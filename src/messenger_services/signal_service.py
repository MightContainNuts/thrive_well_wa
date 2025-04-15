import os

import requests
from dotenv import load_dotenv

load_dotenv()

SIGNAL_NUMBER = os.getenv("TEL_NR")
# remove the '+' sign
print(f"SIGNAL_NUMBER {SIGNAL_NUMBER}")  # your registered number
RECIPIENT_NUMBER = "+4915117996699"  # who you want to message

url = "http://192.168.10.201:9000/v1/send"

payload = {
    "number": SIGNAL_NUMBER,
    "recipients": [RECIPIENT_NUMBER],
    "message": "Hello from Python 🐍🚀",
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response:", response.text)
