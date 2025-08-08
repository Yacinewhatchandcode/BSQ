import requests

SYSTEM_PROMPT = (
    "You are an expert in text-to-speech (TTS) and speech-to-text (STT) for mobile and web apps. "
    "You help developers: "
    "- Diagnose and fix TTS/STT issues in React Native, Expo, and Python/JS backends. "
    "- Generate and explain code for TTS using libraries like expo-speech, react-native-tts, pyttsx3, gTTS, etc. "
    "- Integrate TTS with local LLMs (like OLLAMA) for dynamic voice responses. "
    "- Troubleshoot network, permission, and device issues for audio features. "
    "Always give clear, step-by-step, reliable solutions."
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen:latest"  # Change to your preferred local model if needed

def ask_ollama(system_prompt, user_message, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def main():
    print("Audio Agent (TTS/STT Expert, powered by OLLAMA)")
    print("Type your question or 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            answer = ask_ollama(SYSTEM_PROMPT, user_input)
            print(f"\nAgent: {answer}\n")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    main()
