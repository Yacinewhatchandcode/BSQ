import os
import json
from typing import Dict, Any, List
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class QwenAgent:
    def __init__(self, model_name: str = "qwen3-7b-instruct"):
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = """You are Qwen3-7B-Instruct, an advanced AI agent. Your capabilities include:
- Natural language understanding and generation
- Code analysis and generation
- Problem solving
- Creative writing
- Task planning and execution

You are helpful, honest, and focused on providing accurate and useful responses."""
        
    def chat(self, message: str) -> str:
        """Process a message and return the agent's response."""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Prepare the messages for the model
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history
            
            # Get response from the model
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False
            )
            
            # Extract the response content
            agent_response = response['message']['content']
            
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

def main():
    # Initialize the agent
    agent = QwenAgent()
    
    console.print(Panel.fit(
        "[bold green]Qwen3-7B-Instruct Agent[/bold green]\n"
        "Type 'exit' to quit, 'clear' to clear history, or 'history' to view conversation history.",
        title="Welcome"
    ))
    
    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold blue]You:[/bold blue] ")
            
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
            
            # Get agent response
            response = agent.chat(user_input)
            
            # Display response
            console.print("\n[bold green]Agent:[/bold green]")
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    main()
