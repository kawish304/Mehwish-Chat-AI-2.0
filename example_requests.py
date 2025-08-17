import requests

BASE_URL = "http://127.0.0.1:8000"

# Test Groq LLaMA
chat_payload = {
    "message": "Assalamualaikum! Tell me about Tawheed.",
    "language": "en",           # en, ur, roman_urdu, ar
    "model": "llama3-70b-8192", # fixed LLaMA model
    "temperature": 0.7
}

response = requests.post(f"{BASE_URL}/api/chat", json=chat_payload)

print("Groq LLaMA Response:", response.json())
