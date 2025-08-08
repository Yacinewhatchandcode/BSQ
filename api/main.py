#!/usr/bin/env python3
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import hashlib

app = FastAPI(title="Spiritual Quest API (Vercel)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPIRITUAL_RESPONSES = [
    "O Son of Being! Love Me, that I may love thee. If thou lovest Me not, My love can in no wise reach thee.",
    "O Friend! In the garden of thy heart plant naught but the rose of love, and from the nightingale of affection and desire loathe not to turn away.",
    "O Son of Man! For everything there is a sign. The sign of love is fortitude in My decree and patience in My trials.",
    "O Children of Men! Know ye not why We created you all from the same dust? That no one should exalt himself over the other.",
    "O Son of Spirit! Noble have I created thee, yet thou hast abased thyself. Rise then unto that for which thou wast created.",
    "O Son of Being! Thy heart is My home; sanctify it for My descent.",
]


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": "Vercel Serverless",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/chat")
async def chat(message: str = Form(...)):
    try:
        msg = (message or "").lower()
        if any(k in msg for k in ["love", "beloved"]):
            response = SPIRITUAL_RESPONSES[0]
        elif any(k in msg for k in ["friend", "friendship"]):
            response = SPIRITUAL_RESPONSES[1]
        elif any(k in msg for k in ["sign", "patience", "trial"]):
            response = SPIRITUAL_RESPONSES[2]
        elif any(k in msg for k in ["children", "dust", "equality"]):
            response = SPIRITUAL_RESPONSES[3]
        else:
            h = int(hashlib.md5(message.encode()).hexdigest(), 16)
            response = SPIRITUAL_RESPONSES[h % len(SPIRITUAL_RESPONSES)]
        return {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "source": "The Hidden Words of Bahá'u'lláh",
        }
    except Exception as e:
        return {
            "message": message,
            "response": "I am here to help. Please try again.",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


