from fastapi import FastAPI, Request, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import qrcode
from io import BytesIO
import base64
import json
from rag_agent import SpiritualGuideAgent
from bahai_ux_designer_agent import BahaiUXDesignerAgent
from bahai_mcp_integration import BahaiMCPIntegration
from llm_config import get_llm_config, set_primary_provider, LLMProvider
import asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import requests
from typing import Optional
import time
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent
logger.debug(f"Base directory: {BASE_DIR}")

app = FastAPI(title="Spiritual Quest")

# Mount static files
static_dir = str(BASE_DIR / "static")
logger.debug(f"Static directory: {static_dir}")
if not os.path.exists(static_dir):
    logger.error(f"Static directory does not exist: {static_dir}")
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = str(BASE_DIR / "templates")
logger.debug(f"Templates directory: {templates_dir}")
if not os.path.exists(templates_dir):
    logger.error(f"Templates directory does not exist: {templates_dir}")
    os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Initialize agents and integrations
rag_agent = SpiritualGuideAgent()
ux_designer_agent = BahaiUXDesignerAgent()
mcp_integration = BahaiMCPIntegration()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
model = whisper.load_model("base")

# WebSocket connection settings
WEBSOCKET_TIMEOUT = 1800  # 30 minutes for spiritual conversations
KEEPALIVE_INTERVAL = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

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

    def update_ping(self, client_id: str):
        self.last_ping[client_id] = time.time()

    def check_timeouts(self):
        current_time = time.time()
        for client_id in list(self.last_ping.keys()):
            if current_time - self.last_ping[client_id] > WEBSOCKET_TIMEOUT:
                logger.info(f"Client {client_id} timed out")
                self.disconnect(client_id)

manager = WebSocketManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    logger.debug("Home endpoint called - serving elegant Baha'i manuscript")
    try:
        return templates.TemplateResponse("elegant_bahai_manuscript.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering elegant manuscript template: {str(e)}")
        raise

@app.get("/simple", response_class=HTMLResponse)
async def simple_interface(request: Request):
    """Serve the simple spiritual quest interface"""
    logger.debug("Simple interface endpoint called")
    try:
        return templates.TemplateResponse("spiritual_quest.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering simple template: {str(e)}")
        raise

@app.get("/test", response_class=HTMLResponse)
async def test_interface(request: Request):
    """Serve the test chat interface for debugging"""
    logger.debug("Test interface endpoint called")
    try:
        return templates.TemplateResponse("test_chat.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering test template: {str(e)}")
        raise

@app.get("/qr")
async def generate_qr():
    # Generate QR code for the local server
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data("http://localhost:8000")
    qr.make(fit=True)
    
    # Create QR code image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"qr_code": f"data:image/png;base64,{img_str}"}

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...)):
    """API endpoint for chat without WebSocket"""
    try:
        response = rag_agent.chat(message)
        return {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Chat API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/design")
async def design_endpoint(request: str = Form(...), design_type: str = Form(default="interface")):
    """API endpoint for Baha'i UX Designer Agent"""
    try:
        if design_type == "interface":
            result = ux_designer_agent.design_interface(request)
        elif design_type == "quote":
            # Extract quote and author if provided
            parts = request.split(" - ")
            quote = parts[0].strip('"')
            author = parts[1] if len(parts) > 1 else "Bahá'u'lláh"
            result = ux_designer_agent.design_quote_display(quote, author)
        else:
            result = ux_designer_agent.chat(request)
        
        return {
            "request": request,
            "design_type": design_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Design API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/deploy")
async def deploy_endpoint(design_request: str = Form(...)):
    """Deploy a Baha'i interface to Vercel using MCP integration"""
    try:
        result = mcp_integration.create_and_deploy_interface(design_request)
        return {
            "design_request": design_request,
            "deployment": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Deploy API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/deployments")
async def list_deployments():
    """List all deployments"""
    try:
        deployments = mcp_integration.list_deployments()
        return {
            "deployments": deployments,
            "count": len(deployments),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Deployments API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/llm/providers")
async def get_llm_providers():
    """Get available LLM providers"""
    try:
        config = get_llm_config()
        providers = config.get_available_providers()
        return {
            "providers": providers,
            "current_provider": rag_agent.edge_encoder.primary_provider.value,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"LLM providers API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/set_provider")
async def set_llm_provider(provider: str = Form(...)):
    """Set the primary LLM provider"""
    try:
        # Convert string to enum
        provider_enum = LLMProvider(provider)
        set_primary_provider(provider_enum)
        
        # Update the agent's provider too
        rag_agent.edge_encoder.primary_provider = provider_enum
        
        return {
            "message": f"LLM provider set to {provider}",
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
    except Exception as e:
        logger.error(f"Set LLM provider API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
  
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Transcribe the audio
        result = model.transcribe(temp_file_path)
        transcribed_text = result["text"]

        # Forward the transcribed text to the agent system (Coordinator)
        coordinator_url = "http://localhost:8002/task"
        agent_payload = {
            "text": transcribed_text,
            "context": {},
            "target_agent": "orchestrator"
        }
        agent_response = requests.post(coordinator_url, json=agent_payload, timeout=30)
        agent_data = agent_response.json()

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return {
            "text": transcribed_text,
            "agent_response": agent_data.get("response", ""),
            "status": agent_data.get("status", "error"),
            "error": agent_data.get("error")
        }
    except Exception as e:
        return {"error": str(e)}

async def keepalive_task():
    while True:
        try:
            manager.check_timeouts()
            await manager.broadcast(json.dumps({"type": "ping"}))
            await asyncio.sleep(KEEPALIVE_INTERVAL)
        except Exception as e:
            print(f"Error in keepalive task: {str(e)}")
            await asyncio.sleep(1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"New WebSocket connection attempt from {client_id}")
    
    try:
        await manager.connect(websocket)
        
        # Start keepalive task if not already running
        if not hasattr(app, 'keepalive_task'):
            app.keepalive_task = asyncio.create_task(keepalive_task())
        
        while True:
            try:
                # Set a timeout for receiving messages
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=WEBSOCKET_TIMEOUT
                )
                
                print(f"Received message from {client_id}: {message}")
                
                # Update last ping time
                manager.update_ping(client_id)
                
                # Handle ping messages
                if message == "ping":
                    await manager.send_message(client_id, json.dumps({"type": "pong"}))
                    continue
                
                # Process message with RAG agent
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(rag_agent.chat, message),
                        timeout=30  # 30 second timeout for RAG processing
                    )
                except asyncio.TimeoutError:
                    response = "I apologize, but the request took too long to process. Please try again."
                except Exception as e:
                    print(f"Error in RAG processing: {str(e)}")
                    response = "I apologize, but there was an error processing your request. Please try again."
                
                print(f"Sending response to {client_id}: {response}")
                
                # Send response back to client
                await manager.send_message(
                    client_id,
                    json.dumps({
                        "type": "response",
                        "content": response
                    })
                )
                
            except asyncio.TimeoutError:
                print(f"Timeout waiting for message from {client_id}")
                break
            except WebSocketDisconnect:
                print(f"Client {client_id} disconnected")
                break
            except Exception as e:
                print(f"Error processing message from {client_id}: {str(e)}")
                await manager.send_message(
                    client_id,
                    json.dumps({
                        "type": "error",
                        "content": "An error occurred while processing your message. Please try again."
                    })
                )
    
    except Exception as e:
        print(f"Unexpected error with client {client_id}: {str(e)}")
    finally:
        manager.disconnect(client_id)
        print(f"Connection handler closed for {client_id}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=WEBSOCKET_TIMEOUT,
        timeout_graceful_shutdown=30
    )
