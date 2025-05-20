from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import Optional
import uvicorn

app = FastAPI(title="Audio Agent Service")

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
OLLAMA_MODEL = "qwen:latest"

class AudioRequest(BaseModel):
    text: str
    context: Optional[dict] = None

class AudioResponse(BaseModel):
    response: str
    status: str
    error: Optional[str] = None

def ask_ollama(system_prompt: str, user_message: str, model: str = OLLAMA_MODEL) -> str:
    try:
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=AudioResponse)
async def process_audio(request: AudioRequest):
    try:
        response = ask_ollama(SYSTEM_PROMPT, request.text)
        return AudioResponse(
            response=response,
            status="success"
        )
    except Exception as e:
        return AudioResponse(
            response="",
            status="error",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 