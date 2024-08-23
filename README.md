
Git Clone the project to get started and install the requirements.txt.

This is a basic guide on local set-up and usage. Documentation for deployment will be available soon.

This framework is powered by a fine tuned version of DeBERTa-v3 by ProtectAI:
https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2

Run the program with:

``` python
python qualia.py
```

Different Usage Commands:

![image](https://github.com/user-attachments/assets/7af4006f-45ec-41bc-8262-4d17b2518300)

A test script is provided to test each function upon start-up.

In a terminal run:
```python
python qualia.py --server

or 

python qualia.py --server --cloud
```

In a new terminal run:

```python
python test.py
```

Feel free to edit the test prompts if need be. Full Documentation to release in October.
