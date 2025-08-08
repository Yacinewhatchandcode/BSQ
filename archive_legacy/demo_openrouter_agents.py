#!/usr/bin/env python3
"""
Demo script showing OpenRouter integration with multiple agents
"""

import requests
import json
import os
import time

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterAgent:
    """Simple agent class for OpenRouter integration"""
    def __init__(self, name: str, role: str, model: str = "openrouter/horizon-beta"):
        self.name = name
        self.role = role
        self.model = model
        self.api_key = OPENROUTER_API_KEY
        self.url = OPENROUTER_URL
        
    def chat(self, message: str, system_prompt: str = None) -> str:
        """Send a message to the agent"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": self.model,
            "messages": messages,
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
    
    def __str__(self):
        return f"{self.name} ({self.role}) using {self.model}"

def demo_multi_agent_system():
    """Demonstrate a multi-agent system using OpenRouter"""
    print("🤖 Multi-Agent System with OpenRouter Integration")
    print("=" * 60)
    
    # Create agents
    mobile_agent = OpenRouterAgent(
        name="Mobile Expert",
        role="Mobile UI Specialist",
        model="openrouter/horizon-beta"
    )
    
    backend_agent = OpenRouterAgent(
        name="Backend Engineer", 
        role="API Validator",
        model="openrouter/horizon-beta"
    )
    
    audio_agent = OpenRouterAgent(
        name="Audio Handler",
        role="TTS Specialist", 
        model="openrouter/horizon-beta"
    )
    
    print(f"📱 {mobile_agent}")
    print(f"🔧 {backend_agent}")
    print(f"🎵 {audio_agent}")
    print()
    
    # Simulate agent workflow
    print("🔄 Simulating agent workflow...")
    print("-" * 40)
    
    # Step 1: Mobile agent detects TTS request
    mobile_prompt = "A user wants to convert 'Hello world' to speech. Should we proceed with TTS?"
    mobile_system = "You are a mobile UI specialist. Analyze user requests and determine if TTS is needed."
    
    print(f"📱 {mobile_agent.name} analyzing request...")
    mobile_response = mobile_agent.chat(mobile_prompt, mobile_system)
    print(f"Response: {mobile_response[:100]}...")
    print()
    
    # Step 2: Backend agent validates API
    backend_prompt = "Validate if we have a working TTS API endpoint for the text 'Hello world'"
    backend_system = "You are a backend engineer. Validate API endpoints and suggest fixes if needed."
    
    print(f"🔧 {backend_agent.name} validating API...")
    backend_response = backend_agent.chat(backend_prompt, backend_system)
    print(f"Response: {backend_response[:100]}...")
    print()
    
    # Step 3: Audio agent processes TTS
    audio_prompt = "Generate TTS audio for 'Hello world' and provide implementation details"
    audio_system = "You are an audio specialist. Handle TTS requests and provide technical guidance."
    
    print(f"🎵 {audio_agent.name} processing TTS...")
    audio_response = audio_agent.chat(audio_prompt, audio_system)
    print(f"Response: {audio_response[:100]}...")
    print()
    
    print("✅ Multi-agent workflow completed!")
    print("=" * 60)

def demo_horizon_beta_only():
    """Demonstrate Horizon Beta capabilities"""
    print("\n🚀 Horizon Beta Capabilities Demo")
    print("=" * 40)
    
    test_prompts = [
        "Explain the benefits of using multiple AI models in a single system.",
        "What are the key features of Horizon Beta?",
        "How can Horizon Beta help with software development?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}: {prompt[:50]}...")
        
        agent = OpenRouterAgent(
            name="Horizon Beta Agent",
            role="General Assistant",
            model="openrouter/horizon-beta"
        )
        
        response = agent.chat(prompt)
        print(f"✅ Horizon Beta: {response[:150]}...")
        time.sleep(1)  # Rate limiting

if __name__ == "__main__":
    print("🚀 OpenRouter Multi-Agent System Demo")
    print("=" * 60)
    
    # Demo 1: Multi-agent workflow
    demo_multi_agent_system()
    
    # Demo 2: Horizon Beta capabilities
    demo_horizon_beta_only()
    
    print("\n🎉 Demo completed!")
    print("\n💡 To use this in your project:")
    print("   1. Set LLM_PROVIDER=openrouter")
    print("   2. Use OpenRouterAgent class in your agents")
    print("   3. All agents now use Horizon Beta for consistency") 