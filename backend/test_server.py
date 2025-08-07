#!/usr/bin/env python3
"""
Simple test server to verify FastAPI setup
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test Server", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Starting test server on port 8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080)
