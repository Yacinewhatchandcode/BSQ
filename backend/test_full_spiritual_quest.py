#!/usr/bin/env python3
"""
Comprehensive test script for the Spiritual Quest UI
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

async def test_home_page():
    """Test that the home page loads correctly"""
    print("\n1. Testing Home Page...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/") as resp:
            if resp.status == 200:
                text = await resp.text()
                if "Spiritual Quest" in text:
                    print("✅ Home page loads correctly with Spiritual Quest UI")
                else:
                    print("❌ Home page loads but doesn't contain Spiritual Quest")
            else:
                print(f"❌ Home page failed with status: {resp.status}")

async def test_websocket_connection():
    """Test WebSocket connection"""
    print("\n2. Testing WebSocket Connection...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://localhost:8000/ws") as ws:
                print("✅ WebSocket connected successfully")
                
                # Test ping
                await ws.send_str("ping")
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "pong":
                        print("✅ Ping/pong works correctly")
                
                # Test chat message
                await ws.send_str("Hello, can you share a quote about peace?")
                msg = await asyncio.wait_for(ws.receive(), timeout=15)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if "response" in data:
                        print(f"✅ Chat response received: {data['response'][:100]}...")
                
                await ws.close()
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

async def test_api_chat():
    """Test the API chat endpoint"""
    print("\n3. Testing API Chat Endpoint...")
    async with aiohttp.ClientSession() as session:
        # Test normal conversation
        params = {
            "message": "Hello, how are you today?"
        }
        async with session.post(f"{BASE_URL}/api/chat", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✅ Normal chat works: {data['response'][:100]}...")
            else:
                print(f"❌ Chat API failed with status: {resp.status}")
        
        # Test spiritual quote request
        params = {
            "message": "Can you share a spiritual quote about love?"
        }
        async with session.post(f"{BASE_URL}/api/chat", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✅ Spiritual quote works: {data['response'][:100]}...")

async def test_audio_transcription():
    """Test audio transcription endpoint"""
    print("\n4. Testing Audio Transcription...")
    print("⚠️  Note: This would require an actual audio file to test properly")
    print("   The endpoint is available at /transcribe")
    print("   It accepts audio files and returns transcribed text")

def print_summary():
    """Print test summary"""
    print("\n" + "="*50)
    print("SPIRITUAL QUEST UI TEST SUMMARY")
    print("="*50)
    print("\n✨ Features Available:")
    print("1. Spiritual Quest UI at http://127.0.0.1:8000/")
    print("2. WebSocket connection for real-time chat")
    print("3. API endpoint for chat at /api/chat")
    print("4. Audio transcription at /transcribe")
    print("5. Copy buttons on all messages (hover to see)")
    print("6. Microphone button for voice input")
    print("\n📝 Voice Recording Instructions:")
    print("1. Click the 🎤 button to start recording")
    print("2. Speak your message")
    print("3. Click ⏹️ to stop recording")
    print("4. The transcribed text will appear in the input field")
    print("5. You can edit or copy the text before sending")
    print("\n💡 Tips:")
    print("- Hover over any message to see the copy button")
    print("- The transcribed text is automatically selected for easy copying")
    print("- WebSocket provides real-time responses")
    print("- The system remembers context for natural conversations")

async def main():
    print("Starting Spiritual Quest UI Tests...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    await test_home_page()
    await test_websocket_connection()
    await test_api_chat()
    await test_audio_transcription()
    
    print_summary()

if __name__ == "__main__":
    asyncio.run(main())