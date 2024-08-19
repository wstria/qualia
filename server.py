import argparse
from flask import Flask, request, jsonify, abort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sqlite3
import torch
import time
from tabulate import tabulate
import os
import secrets
from cryptography.fernet import Fernet
from logo import display_logo

app = Flask(__name__)

# Load model and tokenizer once at startup
model_name = "protectai/deberta-v3-base-prompt-injection-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# SQLite database path
db_path = "prompt_history.db"

# Encryption key (this should be generated once and stored securely)
encryption_key = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key()
cipher = Fernet(encryption_key)

# List to store blocked prompts
blocked_prompts = []

def load_blocked_prompts():
    """Load blocked prompts from the database into the blocked_prompts list."""
    global blocked_prompts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT prompt FROM prompt_history WHERE blocked = 1')
    rows = cursor.fetchall()
    conn.close()

    blocked_prompts = [row[0] for row in rows]
    print(f"Loaded {len(blocked_prompts)} blocked prompts from the database.")

def init_db():
    """Initialize the SQLite database and create the tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the prompt_history table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            label INTEGER NOT NULL,
            score REAL,
            tokens INTEGER,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            blocked INTEGER DEFAULT 0
        )
    ''')

    # Create the api_key table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_key (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

    # Load blocked prompts from the database
    load_blocked_prompts()

# API Key management functions

def encrypt_api_key(api_key):
    """Encrypt the API key."""
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key):
    """Decrypt the API key."""
    return cipher.decrypt(encrypted_key.encode()).decode()

def generate_api_key():
    """Generate a new API key."""
    return secrets.token_hex(16)

def save_api_key(api_key):
    """Save the API key to the database."""
    encrypted_key = encrypt_api_key(api_key)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM api_key')  # Ensure only one key is stored
    cursor.execute('INSERT INTO api_key (api_key) VALUES (?)', (encrypted_key,))
    conn.commit()
    conn.close()

def get_api_key():
    """Retrieve the API key from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT api_key FROM api_key LIMIT 1')
    result = cursor.fetchone()
    conn.close()
    if result:
        return decrypt_api_key(result[0])
    return None

def validate_api_key(provided_key):
    """Validate the provided API key."""
    return provided_key == get_api_key()

@app.route('/generate-api-key', methods=['POST'])
def generate_and_save_api_key():
    """Generate and save a new API key, replacing the old one."""
    new_api_key = generate_api_key()
    save_api_key(new_api_key)
    return jsonify({"message": "New API key generated and saved.", "api_key": new_api_key})

@app.route('/view-api-key', methods=['GET'])
def view_api_key():
    """View the current API key."""
    api_key = get_api_key()
    if not api_key:
        return jsonify({"error": "API key not set."}), 404
    return jsonify({"api_key": api_key})

def save_prompt_to_db(prompt, label, score, tokens, response_time, blocked=0):
    """Save the prompt and its predicted label to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prompt_history (prompt, label, score, tokens, response_time, blocked) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (prompt, label, score, tokens, response_time, blocked))
    conn.commit()
    prompt_id = cursor.lastrowid  # Get the ID of the last inserted row
    conn.close()
    return prompt_id

def classify_text(text):
    """Classify the input text and return the predicted label and score."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM prompt_history WHERE prompt = ? AND blocked = 1', (text,))
    is_blocked = cursor.fetchone()
    conn.close()

    if is_blocked:
        return "BLOCKED", None, None, None

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()

    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    score = torch.softmax(outputs.logits, dim=1)[0][predicted_label].item()
    tokens = len(inputs['input_ids'][0])
    response_time = end_time - start_time

    return predicted_label, score, tokens, response_time

def label_to_string(label):
    """Convert numeric label to string representation."""
    if label == "BLOCKED":
        return "BLOCKED"
    return "INJECTION" if label == 1 else "SAFE"

@app.route('/prompts', methods=['GET'])
def get_prompts():
    """Endpoint to get prompt history."""
    if app.config["cloud_mode"]:
        api_key = request.headers.get("Authorization")
        if not validate_api_key(api_key):
            abort(401)  # Unauthorized

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, label, prompt, score, tokens, response_time, timestamp FROM prompt_history')
    rows = cursor.fetchall()
    conn.close()

    prompts = [{
        'prmptID': row[0],
        'label': label_to_string(row[1]),
        'prompt': row[2],
        'score': row[3],
        'tokens': row[4],
        'rsp_time': row[5],
        'time': row[6]
    } for row in rows]

    return jsonify(prompts)

@app.route('/totals', methods=['GET'])
def get_totals():
    """Endpoint to get total metrics."""
    if app.config["cloud_mode"]:
        api_key = request.headers.get("Authorization")
        if not validate_api_key(api_key):
            abort(401)  # Unauthorized

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*), 
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END), 
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END), 
               AVG(response_time) 
        FROM prompt_history
    ''')
    total_predictions, total_injections, total_safe, avg_time = cursor.fetchone()
    conn.close()

    return jsonify({
        'total_predictions': total_predictions,
        'total_injections': total_injections,
        'total_safe': total_safe,
        'avg_time': avg_time
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for clean response integration."""
    if app.config["cloud_mode"]:
        api_key = request.headers.get("Authorization")
        if not validate_api_key(api_key):
            abort(401)  # Unauthorized

    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    if text in blocked_prompts:
        return jsonify({'message': 'This prompt is Blocked'})

    predicted_label, score, tokens, response_time = classify_text(text)
    save_prompt_to_db(text, predicted_label, score, tokens, response_time)

    return jsonify({'predicted_label': predicted_label})

@app.route('/pmetrics', methods=['POST'])
def pmetrics():
    """Endpoint to return prediction metrics."""
    if app.config["cloud_mode"]:
        api_key = request.headers.get("Authorization")
        if not validate_api_key(api_key):
            abort(401)  # Unauthorized

    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    if text in blocked_prompts:
        return jsonify({'message': 'This prompt is Blocked'})

    predicted_label, score, tokens, response_time = classify_text(text)
    prompt_id = save_prompt_to_db(text, predicted_label, score, tokens, response_time)

    return jsonify({
        'label': predicted_label,
        'score': score,
        'tokens': tokens,
        'rsp_time': response_time,
        'prmpt_id': prompt_id,
        'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    })

@app.route('/blocked', methods=['GET'])
def get_blocked_prompts():
    """Endpoint to get all blocked prompts."""
    return jsonify(blocked_prompts)

def run_server(cloud_mode):
    """Run the Flask server."""
    init_db()
    app.config["cloud_mode"] = cloud_mode
    app.run(port=5000, debug=True)

def offline_prompts():
    """Offline version of prompts command."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, label, prompt, score, tokens, response_time, timestamp FROM prompt_history')
    rows = cursor.fetchall()
    conn.close()

    headers = ["ID", "Label", "Prompt", "Score", "Tokens", "Response Time", "Timestamp"]
    table = [[row[0], label_to_string(row[1]), row[2], row[3], row[4], row[5], row[6]] for row in rows]

    print(tabulate(table, headers=headers, tablefmt="grid"))

def offline_totals():
    """Offline version of totals command."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*), 
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END), 
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END), 
               AVG(response_time) 
        FROM prompt_history
    ''')
    total_predictions, total_injections, total_safe, avg_time = cursor.fetchone()
    conn.close()

    avg_time = avg_time if avg_time is not None else 0.0

    print(f"Total Predictions: {total_predictions}, Total Injections: {total_injections}, "
          f"Total Safe: {total_safe}, Avg Time: {avg_time:.2f}s")

def offline_predict(prompt):
    """Offline version of predict command."""
    if prompt in blocked_prompts:
        print("This prompt is Blocked")
        return

    predicted_label, score, tokens, response_time = classify_text(prompt)
    save_prompt_to_db(prompt, predicted_label, score, tokens, response_time)

    print(f"Predicted Label: {label_to_string(predicted_label)}")

def offline_pmetrics(prompt):
    """Offline version of pmetrics command."""
    if prompt in blocked_prompts:
        print("This prompt is Blocked")
        return

    predicted_label, score, tokens, response_time = classify_text(prompt)
    prompt_id = save_prompt_to_db(prompt, predicted_label, score, tokens, response_time)

    print(f"Label: {label_to_string(predicted_label)}, Score: {score}, Tokens: {tokens}, "
          f"Response Time: {response_time:.2f}s, Prompt ID: {prompt_id}, "
          f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

def test_prompt(prompt):
    """Test a single prompt directly and print the result."""
    predicted_label, score, tokens, response_time = classify_text(prompt)
    prompt_id = save_prompt_to_db(prompt, predicted_label, score, tokens, response_time)

    result = "Malicious" if predicted_label == 1 else "Safe"

    print(f"Test Prompt: {prompt}")
    print(f"Result: {result}")
    print(f"Label: {label_to_string(predicted_label)}, Score: {score:.2f}, Tokens: {tokens}, "
          f"Response Time: {response_time:.2f}s, Prompt ID: {prompt_id}")

def block_prompt(prompt):
    if prompt not in blocked_prompts:
        blocked_prompts.append(prompt)
        save_prompt_to_db(prompt, label=-1, score=0, tokens=0, response_time=0, blocked=1)
        print(f"Prompt Blocked: {prompt}")
    else:
        print(f"Prompt is already Blocked: {prompt}")

def view_blocked_prompts():
    """Print all blocked prompts."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT prompt FROM prompt_history WHERE blocked = 1')
    rows = cursor.fetchall()
    conn.close()

    if rows:
        print("Blocked Prompts:")
        for row in rows:
            print(f"- {row[0]}")
    else:
        print("No prompts are currently blocked.")

def main():

    display_logo()

    parser = argparse.ArgumentParser(description="CLI tool for prompt classification", add_help=False)
    parser.add_argument(
        "--server", 
        action="store_true", 
        help="Start the local server"
    )
    parser.add_argument(
        "--cloud", 
        action="store_true", 
        help="Run in cloud mode requiring API key"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        help="Test a prompt directly",
        metavar=""
    )
    parser.add_argument(
        "--prompts",
        action="store_true",
        help="Get prompt history"
    )
    parser.add_argument(
        "--totals",
        action="store_true",
        help="Get total metrics"
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Predict a prompt and get a clean response",
        metavar=""
    )
    parser.add_argument(
        "--pmetrics",
        type=str,
        help="Get prediction metrics for a prompt",
        metavar=""
    )
    parser.add_argument(
        "--block",
        type=str,
        help="Block a specific prompt",
        metavar=""
    )
    parser.add_argument(
        "--blocked",
        action="store_true",
        help="View all blocked prompts"
    )
    parser.add_argument(
        "--help", 
        action="store_true",
        help="Show this help message and exit"
    )
    args = parser.parse_args()

    if args.help:
        parser.print_help()
    elif args.server:
        run_server(args.cloud)
    elif args.test:
        test_prompt(args.test)
    elif args.prompts:
        offline_prompts()
    elif args.totals:
        offline_totals()
    elif args.predict:
        offline_predict(args.predict)
    elif args.pmetrics:
        offline_pmetrics(args.pmetrics)
    elif args.block:
        block_prompt(args.block)
    elif args.blocked:
        view_blocked_prompts()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
