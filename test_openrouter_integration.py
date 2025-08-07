#!/usr/bin/env python3
"""
Test script for OpenRouter integration with Horizon Beta
"""

import os
from langchain_openai import ChatOpenAI

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def test_horizon_beta():
    """Test Horizon Beta via OpenRouter"""
    print("🚀 Testing Horizon Beta via OpenRouter...")
    
    try:
        # Initialize Horizon Beta
        llm = ChatOpenAI(
            model="openrouter/horizon-beta",
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0.7
        )
        
        # Test message
        test_message = "Hello! Can you give me a brief overview of your capabilities?"
        
        print(f"📤 Sending message: {test_message}")
        
        # Get response
        response = llm.invoke(test_message)
        
        print(f"📥 Response from Horizon Beta:")
        print("=" * 50)
        print(response.content)
        print("=" * 50)
        print("✅ OpenRouter integration successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing OpenRouter: {str(e)}")
        return False

def test_model_switching():
    """Test switching between different models"""
    print("\n🔄 Testing model switching...")
    
    models = {
        "horizon-beta": "openrouter/horizon-beta",
        "gpt-4": "openai/gpt-4",
        "claude": "anthropic/claude-3-5-sonnet"
    }
    
    for model_name, model_id in models.items():
        try:
            print(f"\n🧪 Testing {model_name}...")
            
            llm = ChatOpenAI(
                model=model_id,
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0.7
            )
            
            response = llm.invoke("Say hello in one sentence.")
            print(f"✅ {model_name}: {response.content}")
            
        except Exception as e:
            print(f"❌ {model_name}: {str(e)}")

if __name__ == "__main__":
    print("🤖 OpenRouter Integration Test")
    print("=" * 50)
    
    # Test Horizon Beta
    success = test_horizon_beta()
    
    if success:
        # Test other models
        test_model_switching()
    
    print("\n🎉 Test completed!") 