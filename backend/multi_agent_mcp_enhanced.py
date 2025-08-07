from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from typing import List, Dict
import time
import pyttsx3
import os
import requests
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# === LLM CONFIGURATION ===
# Choose your LLM provider: "ollama" or "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Default to Ollama

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model configurations
MODELS = {
    "ollama": {
        "qwen": "qwen:latest",
        "qwen-instruct": "qwen3-7b-instruct"
    },
    "openrouter": {
        "horizon-beta": "openrouter/horizon-beta"
    }
}

class OpenRouterLLM:
    """Custom LLM class for OpenRouter integration"""
    def __init__(self, model_name: str = "openrouter/horizon-beta"):
        self.model_name = model_name
        self.api_key = OPENROUTER_API_KEY
        self.url = OPENROUTER_URL
        
    def __call__(self, prompt: str) -> str:
        """Call the OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
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
        return f"OpenRouterLLM({self.model_name})"

def get_llm(model_name: str = None):
    """Get LLM instance based on provider and model."""
    if LLM_PROVIDER == "ollama":
        model = model_name or MODELS["ollama"]["qwen"]
        return Ollama(model=model)
    
    elif LLM_PROVIDER == "openrouter":
        model = model_name or MODELS["openrouter"]["horizon-beta"]
        return OpenRouterLLM(model)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

# MCP Bus: simple in-memory message/event bus with logs
class MCPBus:
    def __init__(self):
        self.events: List[Dict] = []
        self.logs: List[str] = []

    def send(self, sender, recipient, event, data=None):
        msg = {"from": sender, "to": recipient, "event": event, "data": data, "timestamp": time.time()}
        self.events.append(msg)
        self.logs.append(f"{sender} → {recipient}: {event} | {data}")

    def get_events(self, recipient):
        return [e for e in self.events if e["to"] == recipient]

    def log_error(self, error):
        self.logs.append(f"ERROR: {error}")

    def print_logs(self):
        print("\n--- MCP Bus Logs ---")
        for log in self.logs:
            print(log)

def generate_tts_audio(text, output_dir="tts_output"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"tts_{int(time.time())}.mp3")
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# === AGENT DEFINITIONS ===
# Get LLM instances for different agents
mobile_llm = get_llm()
backend_llm = get_llm()
audio_llm = get_llm()

mobile_agent = Agent(
    name="Mobile Expert",
    role="Mobile UI Specialist",
    goal="Detect UI-to-TTS triggers and request TTS for user text.",
    backstory="You know the mobile codebase and can spot when the UI needs TTS.",
    llm=mobile_llm,
    system_prompt="You watch the /mobile folder, detect UI-to-TTS triggers, and send tts_request(text) to the MCP bus."
)

backend_agent = Agent(
    name="Back-end Engineer",
    role="API Validator",
    goal="Ensure TTS route exists and is valid, or propose a fix.",
    backstory="You know the backend codebase and can validate or fix TTS endpoints.",
    llm=backend_llm,
    system_prompt="You watch /api and /routes, validate TTS route, and respond tts_response_valid or propose a fix."
)

audio_agent = Agent(
    name="Audio Handler",
    role="TTS Simulator",
    goal="Simulate TTS and send audio mock or error log.",
    backstory="You know TTS libraries and can simulate TTS or log errors.",
    llm=audio_llm,
    system_prompt="You check the TTS library, simulate TTS, and send audio mock to front or error log."
)

class MobileFolderHandler(FileSystemEventHandler):
    def __init__(self, bus):
        self.bus = bus

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.txt'):
            with open(event.src_path, 'r') as file:
                text = file.read().strip()
                self.bus.send("Mobile Expert", "Back-end Engineer", "tts_request", {"text": text})

# === AGENT WORKFLOW ===
def run_agents():
    bus = MCPBus()
    
    # Set up the Mobile Expert to watch the /mobile folder
    observer = Observer()
    handler = MobileFolderHandler(bus)
    observer.schedule(handler, path='./mobile', recursive=False)
    observer.start()

    print(f"🚀 Starting multi-agent system with {LLM_PROVIDER.upper()} LLM provider")
    print(f"📱 Mobile Expert: {mobile_agent.llm}")
    print(f"🔧 Back-end Engineer: {backend_agent.llm}")
    print(f"🎵 Audio Handler: {audio_agent.llm}")

    try:
        while True:
            # Step 2: Back-end agent validates TTS route
            events = bus.get_events("Back-end Engineer")
            for event in events:
                if event["event"] == "tts_request" and not event.get("processed"):
                    response = requests.get("http://localhost:8000/tts/validate", params={"text": event['data']['text']})
                    if response.status_code == 200:
                        bus.send("Back-end Engineer", "Audio Handler", "tts_response_valid", {"text": event['data']['text']})
                    else:
                        bus.log_error(f"TTS route validation failed: {response.text}")
                    event["processed"] = True

            # Step 3: Audio agent generates real TTS audio
            events = bus.get_events("Audio Handler")
            for event in events:
                if event["event"] == "tts_response_valid" and not event.get("processed"):
                    audio_file = generate_tts_audio(event['data']['text'])
                    bus.send("Audio Handler", "Mobile Expert", "audio_file", {"audio_file": audio_file, "text": event['data']['text']})
                    event["processed"] = True

            # Step 4: Mobile agent receives audio
            events = bus.get_events("Mobile Expert")
            for event in events:
                if event["event"] == "audio_file" and not event.get("processed"):
                    print(f"📱 Mobile Agent received audio file for: {event['data']['text']} at {event['data']['audio_file']}")
                    event["processed"] = True

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    bus.print_logs()

# === UTILITY FUNCTIONS ===
def switch_llm_provider(provider: str):
    """Switch LLM provider dynamically."""
    global LLM_PROVIDER
    if provider in ["ollama", "openrouter"]:
        LLM_PROVIDER = provider
        print(f"🔄 Switched to {provider.upper()} LLM provider")
        return True
    else:
        print(f"❌ Unsupported provider: {provider}")
        return False

def list_available_models():
    """List all available models for current provider."""
    print(f"\n📋 Available models for {LLM_PROVIDER.upper()}:")
    for model_id, model_name in MODELS[LLM_PROVIDER].items():
        print(f"  • {model_id}: {model_name}")

if __name__ == "__main__":
    print("🤖 Enhanced MCP Multi-Agent System")
    print("=" * 50)
    print(f"Current LLM Provider: {LLM_PROVIDER.upper()}")
    list_available_models()
    print("\n💡 To switch providers, set LLM_PROVIDER environment variable:")
    print("   export LLM_PROVIDER=openrouter  # For Horizon Beta")
    print("   export LLM_PROVIDER=ollama      # For local Qwen")
    print("=" * 50)
    
    run_agents() 