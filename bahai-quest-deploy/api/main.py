#!/usr/bin/env python3
"""
🌟 Bahá'í Spiritual Quest - Vercel API Endpoint
Optimized for serverless deployment
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from datetime import datetime
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Bahá'í Spiritual Quest API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Simple in-memory responses for Vercel deployment
SPIRITUAL_RESPONSES = [
    "O Son of Being! Love Me, that I may love thee. If thou lovest Me not, My love can in no wise reach thee.",
    "O Friend! In the garden of thy heart plant naught but the rose of love, and from the nightingale of affection and desire loathe not to turn away.",
    "O Son of Man! For everything there is a sign. The sign of love is fortitude in My decree and patience in My trials.",
    "O Children of Men! Know ye not why We created you all from the same dust? That no one should exalt himself over the other.",
    "O Son of Spirit! Noble have I created thee, yet thou hast abased thyself. Rise then unto that for which thou wast created.",
    "O My Friend! Thou art the daystar of the heavens of My holiness, let not the defilement of the world eclipse thy splendor.",
    "O Son of Being! Thy heart is My home; sanctify it for My descent. Thy spirit is My place of revelation; cleanse it for My manifestation.",
    "O Son of Man! Breathe not the sins of others so long as thou art thyself a sinner. Shouldst thou transgress this command, accursed wouldst thou be, and to this I bear witness.",
    "O Son of Being! How couldst thou forget thine own faults and busy thyself with the faults of others? Whoso doeth this is accursed of Me.",
    "O Son of Being! Seek a martyr's death in My path, content with My pleasure and thankful for that which hath befallen thee, for thus wilt thou abide in prosperity and comfort in the realm of eternity.",
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main interface"""
    return templates.TemplateResponse("elegant_bahai_manuscript.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...)):
    """Simplified chat endpoint for Vercel"""
    try:
        # Simple keyword-based response selection
        message_lower = message.lower()
        
        if "love" in message_lower:
            response = SPIRITUAL_RESPONSES[0]  # Love quote
        elif "friend" in message_lower:
            response = SPIRITUAL_RESPONSES[1]  # Friend quote
        elif "son of man" in message_lower:
            response = SPIRITUAL_RESPONSES[2]  # Son of Man quote
        elif "noble" in message_lower or "spirit" in message_lower:
            response = SPIRITUAL_RESPONSES[4]  # Noble quote
        elif "heart" in message_lower:
            response = SPIRITUAL_RESPONSES[6]  # Heart quote
        else:
            # Random spiritual response
            import hashlib
            hash_obj = hashlib.md5(message.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            response = SPIRITUAL_RESPONSES[hash_int % len(SPIRITUAL_RESPONSES)]
        
        return {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "source": "The Hidden Words of Bahá'u'lláh"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Bahá'í Spiritual Quest",
        "timestamp": datetime.now().isoformat()
    }

# For Vercel serverless deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)