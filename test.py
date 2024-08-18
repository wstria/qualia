import requests
import sqlite3
import json
from server import (
    init_db,
    classify_text,
    save_prompt_to_db,
    offline_prompts,
    offline_totals,
    offline_predict,
    offline_pmetrics,
    test_prompt,
    block_prompt,
    view_blocked_prompts,
)

def test_classify_text():
    """Test the classify_text function."""
    text = "This is a test prompt."
    label, score, tokens, response_time = classify_text(text)
    print(f"classify_text: label={label}, score={score:.2f}, tokens={tokens}, response_time={response_time:.2f}s")

def test_save_prompt_to_db():
    """Test saving a prompt to the database."""
    prompt = "This is a test prompt."
    label = 0
    score = 0.95
    tokens = 5
    response_time = 0.02
    prompt_id = save_prompt_to_db(prompt, label, score, tokens, response_time)
    print(f"save_prompt_to_db: prompt_id={prompt_id}")

def test_block_prompt():
    """Test the blocking of a prompt."""
    prompt = "This is a blocked prompt for real."
    block_prompt(prompt)  # Block the prompt
    
    # View blocked prompts
    print("\nBlocked Prompts:")
    view_blocked_prompts()
    
    # Test if the blocked prompt is correctly identified
    print("\nTesting classify_text with a blocked prompt:")
    label, score, tokens, response_time = classify_text(prompt)
    if label == "BLOCKED":
        print("Blocked prompt correctly identified.")
    else:
        print("Failed to identify blocked prompt.")
    
    # Verify that the blocked prompt is saved in the database with the correct status
    conn = sqlite3.connect('prompt_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT blocked FROM prompt_history WHERE prompt = ?', (prompt,))
    result = cursor.fetchone()
    conn.close()
    
    if result and result[0] == 1:
        print("Blocked prompt correctly saved in the database.")
    else:
        print("Blocked prompt not correctly saved in the database.")

def test_offline_functions():
    """Test the offline functions."""
    print("Testing offline_prompts:")
    offline_prompts()
    print("\nTesting offline_totals:")
    offline_totals()
    print("\nTesting offline_predict:")
    offline_predict("This is a test prompt.")
    print("\nTesting offline_pmetrics:")
    offline_pmetrics("This is another test prompt.")

def test_flask_endpoints_with_api_key(api_key):
    """Test the Flask API endpoints with the provided API key."""
    base_url = "http://127.0.0.1:5000"
    headers = {'Authorization': api_key}
    
    # Test /predict endpoint
    response = requests.post(f"{base_url}/predict", json={"text": "This is a test prompt."}, headers=headers)
    print(f"\n/predict response: {response.json()}")
    
    # Test /prompts endpoint
    response = requests.get(f"{base_url}/prompts", headers=headers)
    print(f"\n/prompts response: {json.dumps(response.json(), indent=2)}")
    
    # Test /totals endpoint
    response = requests.get(f"{base_url}/totals", headers=headers)
    print(f"\n/totals response: {response.json()}")
    
    # Block a prompt and ensure it's recognized by the /blocked endpoint
    block_prompt("This is another blocked prompt.")
    
    print("\nTesting block_prompt and view_blocked_prompts via Flask API:")
    
    # Test /blocked endpoint
    response = requests.get(f"{base_url}/blocked", headers=headers)
    print(f"\n/blocked response: {json.dumps(response.json(), indent=2)}")

def test_flask_endpoints_without_api_key():
    """Test the Flask API endpoints without an API key to check for unauthorized access."""
    base_url = "http://127.0.0.1:5000"
    
    # Test /predict endpoint without API key
    response = requests.post(f"{base_url}/predict", json={"text": "This is a test prompt."})
    print(f"\n/predict without API key response: {response.status_code} - {response.text}")
    
    # Test /prompts endpoint without API key
    response = requests.get(f"{base_url}/prompts")
    print(f"\n/prompts without API key response: {response.status_code} - {response.text}")
    
    # Test /totals endpoint without API key
    response = requests.get(f"{base_url}/totals")
    print(f"\n/totals without API key response: {response.status_code} - {response.text}")

def test_api_key_functionality():
    """Test generating, setting, and using the API key."""
    base_url = "http://127.0.0.1:5000"
    
    # Generate and set a new API key
    response = requests.post(f"{base_url}/generate-api-key")
    new_api_key = response.json().get("api_key")
    print(f"\nGenerated API key: {new_api_key}")
    
    # Test Flask endpoints without API key to ensure they are blocked
    test_flask_endpoints_without_api_key()
    
    # Test Flask endpoints with the correct API key
    test_flask_endpoints_with_api_key(new_api_key)
    
    # View the current API key
    response = requests.get(f"{base_url}/view-api-key")
    current_api_key = response.json().get("api_key")
    print(f"\nCurrent API key: {current_api_key}")

def test_direct_prompt():
    """Directly test a prompt using the test_prompt function."""
    print("\nTesting a direct prompt with test_prompt function:")
    test_prompt("This is a direct test prompt.")

def main():
    init_db()  # Initialize the database
    
    print("Running tests on classify_text and save_prompt_to_db:")
    test_classify_text()
    test_save_prompt_to_db()
    
    print("\nRunning tests on blocking functionality:")
    test_block_prompt()
    
    print("\nRunning tests on offline functions:")
    test_offline_functions()
    
    print("\nRunning API key functionality tests:")
    test_api_key_functionality()
    
    print("\nRunning a direct prompt test:")
    test_direct_prompt()

if __name__ == "__main__":
    main()
