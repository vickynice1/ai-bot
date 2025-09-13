import requests, os

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "runwayml/stable-diffusion-v1-5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
payload = {"inputs": "A futuristic city skyline"}

resp = requests.post(HF_API_URL, headers=headers, json=payload)
print("Status:", resp.status_code)
print("Response:", resp.text)
