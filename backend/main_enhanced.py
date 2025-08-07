#!/usr/bin/env python3
"""
Enhanced Multi-Agent System Server
Combines FastAPI backend with Horizon Beta agents
"""

from fastapi import FastAPI, Request, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import qrcode
from io import BytesIO
import base64
import json
import asyncio
import uvicorn
import tempfile
import os
import requests
from typing import Optional, Dict, Any
import time
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent
logger.info(f"Base directory: {BASE_DIR}")

app = FastAPI(title="Multi-Agent System", version="1.0.0")

# Mount static files
static_dir = str(BASE_DIR / "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = str(BASE_DIR / "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class HorizonBetaAgent:
    """Horizon Beta agent for OpenRouter integration"""
    def __init__(self, name: str, role: str, system_prompt: str = None):
        self.name = name
        self.role = role
        self.model = "openrouter/horizon-beta"
        self.api_key = OPENROUTER_API_KEY
        self.url = OPENROUTER_URL
        
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = f"You are {name}, a {role}. Provide helpful and accurate responses."
        
    async def chat(self, message: str) -> str:
        """Send a message to Horizon Beta"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize agents
agents = {
    "mobile_expert": HorizonBetaAgent(
        "Mobile Expert", 
        "Mobile UI Specialist",
        "You are a mobile UI specialist. Help with mobile app design, user experience, and interface optimization."
    ),
    "backend_engineer": HorizonBetaAgent(
        "Backend Engineer", 
        "API Validator",
        "You are a backend engineer. Help with API design, validation, and backend system optimization."
    ),
    "audio_handler": HorizonBetaAgent(
        "Audio Handler", 
        "TTS Specialist",
        "You are an audio specialist. Help with text-to-speech, audio processing, and voice synthesis."
    ),
    "rag_agent": HorizonBetaAgent(
        "RAG Agent", 
        "Knowledge Base Specialist",
        "You are a knowledge base specialist. Help with information retrieval, document processing, and knowledge management."
    ),
    "coordinator": HorizonBetaAgent(
        "Coordinator", 
        "System Coordinator",
        "You are a system coordinator. Help orchestrate tasks, manage workflows, and coordinate between different components."
    )
}

class WebSocketManager:
    def __init__(self):
        self.active_connections: dict = {}
        self.last_ping: dict = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = f"{websocket.client.host}:{websocket.client.port}"
        self.active_connections[client_id] = websocket
        self.last_ping[client_id] = time.time()
        logger.info(f"WebSocket connection accepted from {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.last_ping:
            del self.last_ping[client_id]
        logger.info(f"WebSocket connection closed for {client_id}")

    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {str(e)}")
                self.disconnect(client_id)
        return False

    async def broadcast(self, message: str):
        for client_id in list(self.active_connections.keys()):
            await self.send_message(client_id, message)

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with agent interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/agents")
async def list_agents():
    """List all available agents"""
    return {
        "agents": [
            {
                "id": agent_id,
                "name": agent.name,
                "role": agent.role,
                "model": agent.model
            }
            for agent_id, agent in agents.items()
        ]
    }

@app.post("/api/chat")
async def chat_with_agent(agent_id: str, message: str):
    """Chat with a specific agent"""
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents[agent_id]
    response = await agent.chat(message)
    
    # Broadcast to WebSocket clients
    await websocket_manager.broadcast(json.dumps({
        "type": "agent_response",
        "agent": agent_id,
        "message": message,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }))
    
    return {
        "agent": agent_id,
        "message": message,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/tts/generate")
async def generate_tts(text: str):
    """Generate TTS audio (simplified version)"""
    # For now, return a success message
    # In a real implementation, you'd generate actual audio
    return {
        "status": "success",
        "text": text,
        "audio_url": f"/api/tts/audio/{hash(text)}.mp3",
        "message": "TTS generation requested successfully"
    }

@app.get("/api/tts/voices")
async def list_voices():
    """List available TTS voices"""
    return {
        "voices": [
            {"id": "male", "name": "Male Voice", "language": "en"},
            {"id": "female", "name": "Female Voice", "language": "en"},
            {"id": "neutral", "name": "Neutral Voice", "language": "en"}
        ]
    }

@app.post("/api/knowledge/search")
async def search_knowledge(query: str):
    """Search knowledge base"""
    # Use RAG agent for knowledge search
    rag_agent = agents["rag_agent"]
    response = await rag_agent.chat(f"Search for information about: {query}")
    
    return {
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/knowledge/add")
async def add_to_knowledge(content: str, title: str = "Document"):
    """Add content to knowledge base"""
    # In a real implementation, you'd store this in a vector database
    return {
        "status": "success",
        "title": title,
        "message": "Content added to knowledge base"
    }

@app.get("/api/qr")
async def generate_qr():
    """Generate QR code for the server"""
    server_url = "http://localhost:8000"
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(server_url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "qr_code": f"data:image/png;base64,{img_str}",
        "url": server_url
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "chat":
                agent_id = message_data.get("agent_id")
                message = message_data.get("message")
                
                if agent_id in agents:
                    response = await agents[agent_id].chat(message)
                    
                    # Send response back to client
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "agent_id": agent_id,
                        "message": message,
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                    # Broadcast to other clients
                    await websocket_manager.broadcast(json.dumps({
                        "type": "broadcast",
                        "agent_id": agent_id,
                        "message": message,
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }))
            
            elif message_data.get("type") == "ping":
                websocket_manager.last_ping[client_id] = time.time()
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        websocket_manager.disconnect(client_id)

@app.post("/api/workflow/tts")
async def tts_workflow(text: str):
    """Complete TTS workflow using multiple agents"""
    try:
        # Step 1: Mobile agent validates request
        mobile_response = await agents["mobile_expert"].chat(
            f"User wants to convert '{text}' to speech. Should we proceed?"
        )
        
        # Step 2: Backend agent validates API
        backend_response = await agents["backend_engineer"].chat(
            f"Validate TTS API endpoint for text: '{text}'"
        )
        
        # Step 3: Audio agent processes TTS
        audio_response = await agents["audio_handler"].chat(
            f"Generate TTS for text: '{text}'"
        )
        
        # Step 4: Coordinator summarizes
        coordinator_response = await agents["coordinator"].chat(
            f"Summarize the TTS workflow results for text '{text}':\n"
            f"Mobile: {mobile_response}\n"
            f"Backend: {backend_response}\n"
            f"Audio: {audio_response}"
        )
        
        return {
            "workflow": "tts",
            "text": text,
            "steps": {
                "mobile_validation": mobile_response,
                "backend_validation": backend_response,
                "audio_processing": audio_response,
                "coordination": coordinator_response
            },
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"TTS workflow error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflow/knowledge")
async def knowledge_workflow(query: str):
    """Complete knowledge workflow using multiple agents"""
    try:
        # Step 1: RAG agent searches knowledge base
        rag_response = await agents["rag_agent"].chat(
            f"Search knowledge base for: {query}"
        )
        
        # Step 2: Coordinator processes results
        coordinator_response = await agents["coordinator"].chat(
            f"Process and format the knowledge search results for query '{query}':\n{rag_response}"
        )
        
        return {
            "workflow": "knowledge",
            "query": query,
            "steps": {
                "knowledge_search": rag_response,
                "coordination": coordinator_response
            },
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Knowledge workflow error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Multi-Agent System Server...")
    logger.info("Available agents:")
    for agent_id, agent in agents.items():
        logger.info(f"  - {agent_id}: {agent.name} ({agent.role})")
    
    uvicorn.run(
        "main_enhanced:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 