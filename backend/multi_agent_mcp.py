from crewai import Agent, Crew, Task
from langchain_community.llms import Ollama
from typing import List, Dict
import time
import pyttsx3
import os
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

# Use your local OLLAMA LLM (Qwen)
llm = Ollama(model="qwen:latest")

def generate_tts_audio(text, output_dir="tts_output"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"tts_{int(time.time())}.mp3")
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# Define agents
mobile_agent = Agent(
    name="Mobile Expert",
    role="Mobile UI Specialist",
    goal="Detect UI-to-TTS triggers and request TTS for user text.",
    backstory="You know the mobile codebase and can spot when the UI needs TTS.",
    llm=llm,
    system_prompt="You watch the /mobile folder, detect UI-to-TTS triggers, and send tts_request(text) to the MCP bus."
)

backend_agent = Agent(
    name="Back-end Engineer",
    role="API Validator",
    goal="Ensure TTS route exists and is valid, or propose a fix.",
    backstory="You know the backend codebase and can validate or fix TTS endpoints.",
    llm=llm,
    system_prompt="You watch /api and /routes, validate TTS route, and respond tts_response_valid or propose a fix."
)

audio_agent = Agent(
    name="Audio Handler",
    role="TTS Simulator",
    goal="Simulate TTS and send audio mock or error log.",
    backstory="You know TTS libraries and can simulate TTS or log errors.",
    llm=llm,
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

# Simulate agent workflow via MCP Bus
def run_agents():
    bus = MCPBus()
    # Set up the Mobile Expert to watch the /mobile folder
    observer = Observer()
    handler = MobileFolderHandler(bus)
    observer.schedule(handler, path='./mobile', recursive=False)
    observer.start()

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
                    print(f"Mobile Agent received audio file for: {event['data']['text']} at {event['data']['audio_file']}")
                    event["processed"] = True

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    bus.print_logs()

if __name__ == "__main__":
    print("Running MCP multi-agent simulation with real TTS...")
    run_agents()
