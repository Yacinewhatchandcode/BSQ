#!/usr/bin/env python3
"""
🚀 QUICK DEMO CONFIGURATION
Creates a demo configuration for autonomous launch
"""

import json
import time
from pathlib import Path

def create_demo_config():
    """Create demo configuration quickly"""
    config_file = Path(__file__).parent / "spiritual_quest_config.json"
    
    config = {
        "openrouter_api_key": "sk-or-v1-demo",
        "selected_model": "openrouter/horizon-beta",
        "provider": "openrouter",
        "configured_at": time.time(),
        "features": {
            "voice_input": True,
            "golden_cards": True,
            "mobile_support": True,
            "qa_testing": True
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("🎯 Demo configuration created!")
    print("✨ Bahá'í Spiritual Quest ready for autonomous launch!")
    print("\n📊 Configuration Summary:")
    print(f"  🤖 Provider: {config['provider']}")
    print(f"  🧠 Model: {config['selected_model']}")
    print(f"  🎯 Features: {len([k for k, v in config['features'].items() if v])} enabled")
    
    return config

if __name__ == "__main__":
    create_demo_config()