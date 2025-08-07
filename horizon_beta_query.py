import requests
import json

# === CONFIGURATION ===
API_KEY = "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"  # 🔑 Your OpenRouter API key
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "openrouter/horizon-beta"

# === USER PROMPT ===
user_prompt = "What are the benefits of using Horizon Beta over GPT-4?"

# === PAYLOAD ===
payload = {
    "model": MODEL_ID,
    "messages": [
        {"role": "system", "content": "You are Horizon Beta, an elite assistant responding with structured, clear, and concise outputs."},
        {"role": "user", "content": user_prompt}
    ]
}

# === HEADERS ===
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# === SEND REQUEST ===
print("🚀 Querying Horizon Beta...")
response = requests.post(API_URL, headers=headers, json=payload)

# === HANDLE RESPONSE ===
if response.status_code == 200:
    data = response.json()
    message = data['choices'][0]['message']['content']
    print("\n🤖 Horizon Beta says:\n")
    print(message)
    print("\n✅ Query completed successfully!")
else:
    print(f"❌ Request failed with status code {response.status_code}")
    print(response.text)