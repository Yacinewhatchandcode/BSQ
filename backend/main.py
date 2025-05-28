from fastapi import FastAPI, Request, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import qrcode
from io import BytesIO
import base64
import json
from rag_agent import SpiritualGuideAgent
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

# Initialize RAG agent
rag_agent = SpiritualGuideAgent()

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
WEBSOCKET_TIMEOUT = 60  # seconds
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
    logger.debug("Home endpoint called")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
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
