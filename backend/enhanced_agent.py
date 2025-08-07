import os
import json
import requests
from typing import Dict, Any, List
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class EnhancedAgent:
    def __init__(self, provider: str = "ollama", model_name: str = None):
        self.provider = provider
        self.model_name = model_name or self._get_default_model()
        self.conversation_history: List[Dict[str, str]] = []
        
        # OpenRouter configuration
        self.openrouter_api_key = "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"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Set system prompt based on provider
        if provider == "openrouter":
            self.system_prompt = """You are Horizon Beta, an elite AI assistant. Your capabilities include:
- Natural language understanding and generation
- Code analysis and generation
- Problem solving
- Creative writing
- Task planning and execution
- Structured, clear, and concise outputs

You are helpful, honest, and focused on providing accurate and useful responses."""
        else:
            self.system_prompt = """You are Qwen3-7B-Instruct, an advanced AI agent. Your capabilities include:
- Natural language understanding and generation
- Code analysis and generation
- Problem solving
- Creative writing
- Task planning and execution

You are helpful, honest, and focused on providing accurate and useful responses."""
        
    def _get_default_model(self) -> str:
        """Get default model based on provider"""
        if self.provider == "openrouter":
            return "openrouter/horizon-beta"
        else:
            return "qwen3-7b-instruct"
    
    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama API"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"
        
    def chat(self, message: str) -> str:
        """Process a message and return the agent's response."""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Prepare the messages for the model
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history
            
            # Get response from the appropriate provider
            if self.provider == "openrouter":
                agent_response = self._call_openrouter(messages)
            else:
                agent_response = self._call_ollama(messages)
            
            # Add agent response to history
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            
            return agent_response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def switch_provider(self, new_provider: str):
        """Switch between Ollama and OpenRouter"""
        if new_provider in ["ollama", "openrouter"]:
            self.provider = new_provider
            self.model_name = self._get_default_model()
            console.print(f"[green]Switched to {new_provider.upper()} provider with model {self.model_name}[/green]")
        else:
            console.print(f"[red]Unsupported provider: {new_provider}[/red]")

def main():
    # Initialize the agent (default to Ollama)
    agent = EnhancedAgent(provider="ollama")
    
    console.print(Panel.fit(
        f"[bold green]Enhanced Agent - {agent.provider.upper()}[/bold green]\n"
        f"Model: {agent.model_name}\n"
        "Type 'exit' to quit, 'clear' to clear history, 'history' to view conversation history,\n"
        "'switch ollama' or 'switch openrouter' to change providers.",
        title="Welcome"
    ))
    
    while True:
        try:
            # Get user input
            user_input = console.input(f"\n[bold blue]You ({agent.provider}):[/bold blue] ")
            
            # Handle special commands
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'clear':
                agent.clear_history()
                console.print("[yellow]Conversation history cleared.[/yellow]")
                continue
            elif user_input.lower() == 'history':
                history = agent.get_history()
                for msg in history:
                    role = "You" if msg["role"] == "user" else "Agent"
                    console.print(f"\n[bold]{role}:[/bold] {msg['content']}")
                continue
            elif user_input.lower().startswith('switch '):
                provider = user_input.lower().split(' ')[1]
                agent.switch_provider(provider)
                continue
            
            # Get agent response
            response = agent.chat(user_input)
            
            # Display response
            console.print(f"\n[bold green]Agent ({agent.provider}):[/bold green]")
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    main() 