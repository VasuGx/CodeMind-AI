import os
from dotenv import load_dotenv

def test_groq_keys_present():
    load_dotenv()
    assert os.getenv("GROQ_API_KEY_1") is not None, "GROQ_API_KEY_1 is missing"
