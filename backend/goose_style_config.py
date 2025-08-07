#!/usr/bin/env python3
"""
🎯 GOOSE-STYLE CONFIGURATION SYSTEM
Interactive configuration like Goose with OpenRouter integration
"""

import os
import json
import requests
import webbrowser
import time
from pathlib import Path

class GooseStyleConfigurator:
    def __init__(self):
        self.config_file = Path(__file__).parent / "spiritual_quest_config.json"
        
    def welcome(self):
        """Display Goose-style welcome"""
        print("\n" + "─" * 60)
        print("┌   Bahá'í Spiritual Quest Configuration")
        print("│")
        print("│   🌟 Welcome to the autonomous setup!")
        print("│   Let's configure your spiritual AI system.")
        print("│")
        
    def configure_openrouter(self):
        """Configure OpenRouter like Goose"""
        print("◇  How would you like to set up your AI provider?")
        print("│  OpenRouter Login (Recommended)")
        print("│")
        
        choice = input("│  Press Enter to continue with OpenRouter or 'skip' to use demo mode: ").strip().lower()
        
        if choice == 'skip':
            print("│  ⚠️  Using demo mode - limited functionality")
            return self.create_demo_config()
        
        return self.setup_openrouter()
    
    def setup_openrouter(self):
        """Set up OpenRouter authentication"""
        print("│")
        print("◇  Choose authentication method:")
        print("│  ● Manual API Key Entry")
        print("│  ○ Browser Authentication (Like Goose)")
        print("│")
        
        auth_choice = input("│  Enter 'browser' for web auth or press Enter for manual: ").strip().lower()
        
        if auth_choice == 'browser':
            return self.browser_auth()
        else:
            return self.manual_auth()
    
    def browser_auth(self):
        """Browser-based authentication like Goose"""
        print("│")
        print("◐  Opening browser for authentication...")
        
        # Simulate Goose-style browser auth
        auth_url = "https://openrouter.ai/keys"
        
        try:
            webbrowser.open(auth_url)
            print(f"│  Auth URL: {auth_url}")
            print("│  Waiting for you to copy your API key...")
            
            api_key = input("│  Paste your OpenRouter API key here: ").strip()
            
            if self.validate_api_key(api_key):
                print("│  ✓ API key validated successfully!")
                return self.complete_config(api_key)
            else:
                print("│  ❌ Invalid API key. Using fallback mode.")
                return self.create_demo_config()
                
        except Exception as e:
            print(f"│  ⚠️  Browser auth failed: {e}")
            return self.manual_auth()
    
    def manual_auth(self):
        """Manual API key entry"""
        print("│")
        print("◇  Please enter your OpenRouter API key:")
        print("│   (Get one free at: https://openrouter.ai/keys)")
        print("│")
        
        api_key = input("│  API Key: ").strip()
        
        if not api_key:
            print("│  No API key provided. Using demo mode.")
            return self.create_demo_config()
        
        if self.validate_api_key(api_key):
            print("│  ✓ API key validated!")
            return self.complete_config(api_key)
        else:
            print("│  ⚠️  Could not validate key. Continuing anyway...")
            return self.complete_config(api_key)
    
    def validate_api_key(self, api_key):
        """Validate OpenRouter API key"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Test with a simple request
            response = requests.get(
                'https://openrouter.ai/api/v1/models',
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
            
        except:
            return False
    
    def complete_config(self, api_key):
        """Complete configuration with model selection"""
        print("│")
        print("◐  Fetching available models...")
        
        models = self.get_available_models(api_key)
        
        print("│")
        print("◆  Select your preferred model:")
        
        # Display models Goose-style
        preferred_models = [
            "openrouter/horizon-beta",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o", 
            "openai/gpt-4o-mini"
        ]
        
        for i, model in enumerate(preferred_models):
            marker = "●" if i == 0 else "○"
            print(f"│  {marker} {model}")
        
        print("│")
        choice = input("│  Enter number (0-4) or press Enter for Horizon Beta: ").strip()
        
        try:
            selected_model = preferred_models[int(choice)]
        except:
            selected_model = preferred_models[0]
        
        print(f"│  Selected: {selected_model}")
        print("│")
        print("◐  Saving configuration...")
        
        config = {
            "openrouter_api_key": api_key,
            "selected_model": selected_model,
            "provider": "openrouter", 
            "configured_at": time.time(),
            "features": {
                "voice_input": True,
                "golden_cards": True,
                "mobile_support": True,
                "qa_testing": True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set environment variables
        os.environ['OPENROUTER_API_KEY'] = api_key
        os.environ['SELECTED_MODEL'] = selected_model
        
        print("│  ✓ Configuration complete!")
        print("│")
        print("◐  Testing configuration...")
        
        if self.test_configuration(config):
            print("│  ✓ Configuration test passed!")
            print("└  OpenRouter setup complete! Ready for autonomous launch.")
            return config
        else:
            print("│  ⚠️  Configuration test failed, but continuing...")
            print("└  Setup complete with warnings.")
            return config
    
    def create_demo_config(self):
        """Create demo configuration"""
        config = {
            "openrouter_api_key": "demo_mode",
            "selected_model": "demo/spiritual-guide",
            "provider": "demo",
            "configured_at": time.time(),
            "features": {
                "voice_input": False,
                "golden_cards": True,
                "mobile_support": True,
                "qa_testing": False
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("│  Demo configuration created")
        print("└  Ready for autonomous launch in demo mode.")
        return config
    
    def get_available_models(self, api_key):
        """Get available models from OpenRouter"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
            }
            
            response = requests.get(
                'https://openrouter.ai/api/v1/models',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get('data', [])
            
        except:
            pass
        
        return []
    
    def test_configuration(self, config):
        """Test the configuration"""
        try:
            if config['provider'] == 'demo':
                return True
            
            headers = {
                'Authorization': f'Bearer {config["openrouter_api_key"]}',
                'Content-Type': 'application/json'
            }
            
            test_data = {
                "model": config['selected_model'],
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=test_data,
                timeout=15
            )
            
            return response.status_code == 200
            
        except:
            return False
    
    def run_configuration(self):
        """Run the full configuration process"""
        self.welcome()
        
        if self.config_file.exists():
            print("◇  Existing configuration found. Update it?")
            print("│  ○ Yes, reconfigure")
            print("│  ● No, use existing")
            print("│")
            
            update = input("│  Update? (y/N): ").strip().lower()
            
            if update != 'y':
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print("│  Using existing configuration")
                print("└  Configuration loaded successfully!")
                return config
        
        config = self.configure_openrouter()
        
        print("\n" + "─" * 60)
        print("🎉 Configuration Complete!")
        print("✨ Your Bahá'í Spiritual Quest system is ready!")
        print("🚀 Run 'python autonomous_launcher.py' to start everything")
        
        return config

def main():
    """Main configuration entry point"""
    configurator = GooseStyleConfigurator()
    config = configurator.run_configuration()
    
    # Print summary
    print(f"\n📊 Configuration Summary:")
    print(f"  🤖 Provider: {config['provider']}")
    print(f"  🧠 Model: {config['selected_model']}")
    print(f"  🎯 Features: {len([k for k, v in config['features'].items() if v])} enabled")
    
    return config

if __name__ == "__main__":
    main()