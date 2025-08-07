#!/usr/bin/env python3
"""
🌟 Bahá'í Spiritual Quest - Production API Endpoint
Optimized for Vercel serverless deployment with MCP integration
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from datetime import datetime
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Bahá'í Spiritual Quest API",
    description="Sacred digital space for exploring The Hidden Words",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Spiritual responses from The Hidden Words
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
    "O Son of Man! Rejoice in the gladness of thine heart, that thou mayest be worthy to meet Me and to mirror forth My beauty.",
    "O Friend! Thou art the light of the world. Let not the shadows of vain glory obscure thy brightness.",
    "O Son of Spirit! The best beloved of all things in My sight is Justice; turn not away therefrom if thou desirest Me, and neglect it not that I may confide in thee.",
    "O Children of Dust! Tell the rich of the midnight sighing of the poor, lest heedlessness lead them into the path of destruction."
]

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main spiritual interface"""
    try:
        # Try to read the HTML content
        html_path = Path(__file__).parent.parent / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding='utf-8')
        else:
            # Fallback content
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>The Hidden Words - Sacred Digital Manuscript</title>
                <meta charset="utf-8">
                <style>
                    body { 
                        font-family: 'Cinzel', serif; 
                        background: #0B0E1A; 
                        color: #E6D4A3; 
                        text-align: center; 
                        padding: 50px;
                    }
                    h1 { font-size: 3em; margin-bottom: 30px; }
                    p { font-size: 1.2em; line-height: 1.6; }
                </style>
            </head>
            <body>
                <h1>كلمات مخفیه</h1>
                <h2>The Hidden Words - Sacred Digital Manuscript</h2>
                <p>Welcome to this sacred digital space, where the eternal wisdom illuminates our spiritual journey.</p>
                <p>🚀 Successfully deployed with MCP Vercel Integration</p>
            </body>
            </html>
            """
    except Exception as e:
        return f"<h1>Bahá'í Spiritual Quest</h1><p>Status: Online</p><p>Error: {str(e)}</p>"

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...)):
    """Spiritual guidance endpoint with enhanced responses"""
    try:
        message_lower = message.lower()
        
        # Enhanced response selection based on keywords
        if any(word in message_lower for word in ["love", "loving", "beloved"]):
            response = SPIRITUAL_RESPONSES[0]
        elif any(word in message_lower for word in ["friend", "friendship", "companion"]):
            response = SPIRITUAL_RESPONSES[1] 
        elif any(word in message_lower for word in ["sign", "signs", "patience", "trial"]):
            response = SPIRITUAL_RESPONSES[2]
        elif any(word in message_lower for word in ["children", "equality", "dust", "created"]):
            response = SPIRITUAL_RESPONSES[3]
        elif any(word in message_lower for word in ["noble", "spirit", "spiritual", "created"]):
            response = SPIRITUAL_RESPONSES[4]
        elif any(word in message_lower for word in ["daystar", "light", "brightness", "glory"]):
            response = SPIRITUAL_RESPONSES[5]
        elif any(word in message_lower for word in ["heart", "home", "revelation", "manifestation"]):
            response = SPIRITUAL_RESPONSES[6]
        elif any(word in message_lower for word in ["justice", "just", "beloved", "sight"]):
            response = SPIRITUAL_RESPONSES[12]
        elif any(word in message_lower for word in ["rich", "poor", "wealth", "poverty"]):
            response = SPIRITUAL_RESPONSES[13]
        elif any(word in message_lower for word in ["rejoice", "gladness", "joy", "happiness"]):
            response = SPIRITUAL_RESPONSES[10]
        else:
            # Select based on message hash for consistency
            import hashlib
            hash_obj = hashlib.md5(message.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            response = SPIRITUAL_RESPONSES[hash_int % len(SPIRITUAL_RESPONSES)]
        
        return {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "source": "The Hidden Words of Bahá'u'lláh",
            "deployment": "MCP Vercel Integration v2.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spiritual guidance error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "service": "Bahá'í Spiritual Quest",
        "version": "2.0.0",
        "deployment": "MCP Vercel Integration",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Golden Quote Cards",
            "Persian Text Support", 
            "Spiritual Guidance AI",
            "Starfield Animation",
            "Mobile Responsive"
        ]
    }

@app.get("/api/quotes")
async def get_random_quote():
    """Get a random spiritual quote"""
    import random
    quote = random.choice(SPIRITUAL_RESPONSES)
    return {
        "quote": quote,
        "source": "The Hidden Words of Bahá'u'lláh",
        "timestamp": datetime.now().isoformat()
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
