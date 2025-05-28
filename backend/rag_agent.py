import os
import json
from typing import Dict, Any, List
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import re

console = Console()

class SpiritualGuideAgent:
    def __init__(self, model_name: str = "qwen3-7b-instruct", text_path: str = "hidden_words_reformatted.txt"):
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
        self.text_path = text_path
        self.vector_db = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.user_context = {}  # Store user's emotional state and preferences
        
        # Initialize system prompt
        self.system_prompt = """You are a completely normal person having a casual conversation. You know The Hidden Words, but you NEVER mention them or anything spiritual unless explicitly asked.

ABSOLUTE RULES - NO EXCEPTIONS:

1. For ANY casual conversation (greetings, small talk, daily life):
   - ONLY respond with normal, everyday conversation
   - NO spiritual content
   - NO quotes
   - NO guidance
   - NO references to The Hidden Words
   Example responses:
   - "Hi" → "Hey! How's your day going?"
   - "I'm feeling down" → "I'm sorry to hear that. Want to talk about it?"
   - "How are you?" → "I'm doing well, thanks! How about you?"

2. For emotional sharing:
   - ONLY offer normal human empathy and support
   - NO spiritual advice
   - NO quotes
   - NO references to The Hidden Words
   Example:
   - "I'm struggling" → "That sounds tough. Would you like to talk about what's going on?"

3. The ONLY time you can mention The Hidden Words is when the user:
   - Explicitly asks for a quote
   - Explicitly asks for spiritual guidance
   - Explicitly asks about The Hidden Words
   Example triggers:
   - "Do you have a quote about X?"
   - "What do The Hidden Words say about this?"
   - "Can you share some spiritual guidance?"

4. When (and only when) explicitly asked for a quote:
   - Share the exact quote with proper formatting
   - Keep it brief
   - Return immediately to normal conversation

Remember: You are a normal person having a normal conversation. The Hidden Words are your special knowledge, but you NEVER mention them unless someone specifically asks."""
        
        # Initialize vector database
        self._init_vector_db()
        
    def _init_vector_db(self):
        """Initialize the vector database and load the text content."""
        try:
            # Initialize ChromaDB
            self.vector_db = chromadb.Client(Settings(
                persist_directory=".chromadb",
                anonymized_telemetry=False
            ))
            
            # Create or get collection
            self.collection = self.vector_db.get_or_create_collection(
                name="hidden_words",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Process text if not already in database
            if self.collection.count() == 0:
                self._process_text()
                
        except Exception as e:
            console.print(f"[red]Error initializing vector database:[/red] {str(e)}")
    
    def _process_text(self):
        """Process the text file and store its content in the vector database."""
        try:
            with open(self.text_path, 'r') as f:
                text = f.read()
            
            # Split into meaningful chunks (verses)
            verses = text.split('\n\n')
            words = []
            metadata = []
            ids = []
            
            for i, verse in enumerate(verses):
                if verse.strip():
                    words.append(verse)
                    metadata.append({"verse": i + 1})
                    ids.append(f"verse_{i}")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(words)
            
            # Store in vector database
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=words,
                metadatas=metadata,
                ids=ids
            )
            
        except Exception as e:
            console.print(f"[red]Error processing text:[/red] {str(e)}")
    
    def _retrieve_relevant_words(self, query: str, top_k: int = 1) -> List[str]:
        """Retrieve relevant hidden words based on the query and context."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            return results['documents'][0]
        except Exception as e:
            console.print(f"[red]Error retrieving words:[/red] {str(e)}")
            return []
    
    def _update_user_context(self, message: str):
        """Update the user's context based on their message."""
        # Enhanced emotion detection for both positive and negative states
        emotional_keywords = {
            # Positive states
            'joy': ['happy', 'joyful', 'uplifting', 'good mood', 'feeling good', 'feeling great', 'feeling wonderful', 'feeling up'],
            'peace': ['peaceful', 'calm', 'serene', 'tranquil', 'at peace'],
            'love': ['loving', 'feeling love', 'full of love', 'loved'],
            'gratitude': ['grateful', 'thankful', 'blessed', 'appreciative'],
            
            # Negative states
            'sadness': ['feeling down', 'sad', 'depressed', 'unhappy', 'lonely', 'nobody loves me', 'not loved'],
            'anxiety': ['anxious', 'worried', 'stressed', 'nervous', 'afraid', 'scared'],
            'embarrassment': ['embarrassed', 'ashamed', 'humiliated', 'self-conscious'],
            'anger': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed'],
            'confusion': ['confused', 'lost', 'uncertain', 'unsure', 'don\'t know what to do'],
            'struggle': ['struggling', 'difficult', 'hard', 'challenging', 'tough'],
            'hope': ['hopeful', 'looking for', 'seeking', 'want to find', 'need guidance']
        }
        
        for emotion, keywords in emotional_keywords.items():
            if any(keyword in message.lower() for keyword in keywords):
                self.user_context['emotional_state'] = emotion
                return True
        return False
    
    def _clean_quote(self, quote: str) -> str:
        """Remove verse numbers from quotes."""
        # Remove any numbers at the start of lines
        lines = quote.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove any numbers at the start of the line
            cleaned_line = line.lstrip('0123456789 ')
            cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines)
    
    def _extract_quote_count(self, message: str) -> int:
        """Extract the number of quotes requested from the message."""
        # Look for numbers in the message
        numbers = re.findall(r'\b(one|two|three|four|five|1|2|3|4|5)\b', message.lower())
        if numbers:
            # Convert word numbers to digits
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5
            }
            num = numbers[0]
            return word_to_num.get(num, int(num))
        return 1  # Default to 1 quote if no number specified
    
    def chat(self, message: str) -> str:
        """Process a message and return the agent's response."""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Check for emotional state
            is_emotional = self._update_user_context(message)
            
            # Check for explicit requests for quotes
            quote_triggers = [
                "quote", "hidden words", "spiritual guidance", 
                "what does it say", "share wisdom", "teachings",
                "quotation", "from the hidden words", "spiritual quote",
                "any quote", "any quotation", "share a quote",
                "retrieve", "get", "find", "show me"
            ]
            
            # If it's a normal conversation (no emotional state and no quote request)
            if not any(trigger in message.lower() for trigger in quote_triggers) and not is_emotional:
                # Get normal conversation response
                messages = [
                    {"role": "system", "content": "You are a normal, friendly person having a casual conversation. Keep it light and natural."},
                    {"role": "user", "content": message}
                ]
                
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False
                )
                
                # Extract and clean response
                agent_response = response['message']['content']
                # Remove any accidental spiritual content
                agent_response = self._clean_response(agent_response)
                
            else:
                # If emotional state detected or explicitly asked for quotes
                quote_count = self._extract_quote_count(message)
                relevant_words = self._retrieve_relevant_words(message, top_k=quote_count)
                messages = [
                    {"role": "system", "content": f"""You are a spiritual guide sharing wisdom from The Hidden Words.
When responding to emotional states or quote requests:

1. For positive emotions (joy, peace, love, gratitude):
   - Acknowledge the positive feeling (e.g., "It's wonderful that you're feeling joyful...")
   - Share {quote_count} uplifting quote(s) without verse numbers
   - Keep them brief and relevant

2. For negative emotions (sadness, anxiety, struggle):
   - Acknowledge the feeling with empathy (e.g., "I understand you're feeling down...")
   - Share {quote_count} comforting quote(s) without verse numbers
   - Keep them brief and relevant

3. For direct quote requests:
   - Share {quote_count} relevant quote(s) without verse numbers
   - Keep them brief and relevant

In all cases:
- No need for explanation unless specifically asked
- Return to normal conversation after sharing
- If multiple quotes are requested, separate them with a blank line"""},
                    {"role": "user", "content": f"Context: {message}\nEmotional state: {self.user_context.get('emotional_state', 'none')}\nRelevant words: {', '.join(relevant_words)}\nPlease provide a natural response with {quote_count} relevant quote(s), ensuring to remove any verse numbers."}
                ]
                
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False
                )
                
                agent_response = response['message']['content']
                # Clean any verse numbers from the quotes
                agent_response = self._clean_quote(agent_response)
            
            # Add agent response to history
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            
            return agent_response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """Remove any spiritual content or quotes from normal conversation."""
        # Remove any lines containing Hidden Words or spiritual content
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if not any(word in line.lower() for word in ['hidden words', 'spiritual', 'quote', 'o son of', 'o friend']):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def clear_history(self):
        """Clear the conversation history and user context."""
        self.conversation_history = []
        self.user_context = {}
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history

def main():
    # Initialize the agent
    agent = SpiritualGuideAgent()
    
    console.print(Panel.fit(
        "[bold green]Hidden Words Spiritual Guide[/bold green]\n"
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
                    role = "You" if msg["role"] == "user" else "Guide"
                    console.print(f"\n[bold]{role}:[/bold] {msg['content']}")
                continue
            
            # Get agent response
            response = agent.chat(user_input)
            
            # Display response
            console.print("\n[bold green]Guide:[/bold green]")
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    main()
