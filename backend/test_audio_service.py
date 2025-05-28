import requests
import json
from typing import List, Dict
import time

TEST_CASES = [
    {
        "text": "How do I implement TTS in my React Native app?",
        "expected_keywords": ["expo-speech", "react-native-tts", "implementation"]
    },
    {
        "text": "What are common issues with audio recording in Expo?",
        "expected_keywords": ["permissions", "expo-av", "recording"]
    },
    {
        "text": "How can I integrate OLLAMA with TTS?",
        "expected_keywords": ["integration", "OLLAMA", "text-to-speech"]
    }
]

def test_audio_service():
    base_url = "http://localhost:8001"
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health")
        assert health_response.status_code == 200
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return

    # Test each test case
    for i, test_case in enumerate(TEST_CASES, 1):
        try:
            print(f"\nTesting case {i}: {test_case['text']}")
            
            # Send request to process endpoint
            response = requests.post(
                f"{base_url}/process",
                json={"text": test_case["text"]}
            )
            
            # Check response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["response"]
            
            # Check for expected keywords
            response_text = data["response"].lower()
            missing_keywords = [
                keyword for keyword in test_case["expected_keywords"]
                if keyword.lower() not in response_text
            ]
            
            if missing_keywords:
                print(f"⚠️ Missing keywords: {missing_keywords}")
            else:
                print("✅ All expected keywords found")
            
            print(f"Response: {data['response'][:200]}...")
            
        except Exception as e:
            print(f"❌ Test case {i} failed: {str(e)}")
        
        # Add delay between requests
        time.sleep(1)

if __name__ == "__main__":
    print("Starting Audio Service Tests...")
    test_audio_service()
    print("\nTests completed!")
