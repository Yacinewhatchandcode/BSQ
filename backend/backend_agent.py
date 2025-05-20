from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import uvicorn

app = FastAPI(title="Back-end Agent Service")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen:latest"

SYSTEM_PROMPT = (
    "You are a backend expert agent. "
    "You help with API design, database queries, authentication, and backend code. "
    "Always provide clear, step-by-step, reliable solutions."
)

class BackendTaskRequest(BaseModel):
    text: str
    context: Optional[dict] = None
    use_llm: Optional[bool] = False

class BackendTaskResponse(BaseModel):
    response: str
    status: str
    error: Optional[str] = None

def ask_ollama(system_prompt: str, user_message: str, model: str = OLLAMA_MODEL) -> str:
    try:
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\nUser: {user_message}\nBackend Agent:",
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/task", response_model=BackendTaskResponse)
async def handle_backend_task(request: BackendTaskRequest):
    try:
        if request.use_llm:
            response = ask_ollama(SYSTEM_PROMPT, request.text)
            return BackendTaskResponse(
                response=response,
                status="success"
            )
        else:
            # For now, just echo the request and simulate backend logic
            response = f"[Back-end Agent] Received: {request.text}"
            return BackendTaskResponse(
                response=response,
                status="success"
            )
    except Exception as e:
        return BackendTaskResponse(
            response="",
            status="error",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003) 