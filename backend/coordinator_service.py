from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from typing import Optional
import uvicorn

app = FastAPI(title="Coordinator Agent Service")

AUDIO_AGENT_URL = "http://localhost:8001/process"
BACKEND_AGENT_URL = "http://localhost:8003/task"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen:latest"

SYSTEM_PROMPT = (
    "You are a central orchestrator agent. "
    "You decide which agent (audio, backend, mobile, or yourself) should handle a task. "
    "If the user asks a general question, answer directly. "
    "If the task is for another agent, explain your routing decision."
)

class TaskRequest(BaseModel):
    text: str
    context: Optional[dict] = None
    target_agent: Optional[str] = "audio"  # 'audio', 'backend', or 'orchestrator'

class TaskResponse(BaseModel):
    response: str
    status: str
    error: Optional[str] = None

def ask_ollama(system_prompt: str, user_message: str, model: str = OLLAMA_MODEL) -> str:
    try:
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\nUser: {user_message}\nOrchestrator:",
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/task", response_model=TaskResponse)
async def handle_task(request: TaskRequest):
    try:
        if request.target_agent == "orchestrator":
            response = ask_ollama(SYSTEM_PROMPT, request.text)
            return TaskResponse(
                response=response,
                status="success"
            )
        elif request.target_agent == "backend":
            agent_url = BACKEND_AGENT_URL
        else:
            agent_url = AUDIO_AGENT_URL
        agent_response = requests.post(
            agent_url,
            json={"text": request.text, "context": request.context}
        )
        agent_response.raise_for_status()
        data = agent_response.json()
        return TaskResponse(
            response=data.get("response", ""),
            status=data.get("status", "error"),
            error=data.get("error")
        )
    except Exception as e:
        return TaskResponse(
            response="",
            status="error",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 