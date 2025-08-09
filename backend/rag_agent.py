import os
import json
from typing import Dict, Any, List
import requests
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
except Exception:
    class Console:
        def print(self, *args, **kwargs):
            pass
    Panel = object
    Markdown = str
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    class Settings:  # type: ignore
        def __init__(self, *a, **k):
            pass
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore
try:
    import numpy as np
except Exception:
    np = None
import re
import asyncio
from llm_config import EdgeEncoder, LLMProvider, get_edge_encoder
from pathlib import Path

console = Console()

# OpenRouter Configuration (fallback)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

class SpiritualGuideAgent:
    def __init__(self, model_name: str = "openai_gpt5_mini", text_path: str = "hidden_words_reformatted.txt"):
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
        # Resolve text path relative to this file to avoid CWD issues
        base_dir = Path(__file__).resolve().parent
        self.text_path = str(base_dir / text_path) if not os.path.isabs(text_path) else text_path
        self.vector_db = None
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.embedding_model = None
        # Lightweight raw verses for fallback retrieval when embeddings are unavailable
        self.raw_verses: List[str] = self._load_raw_verses(self.text_path)
        self.user_context = {}  # Store user's emotional state and preferences
        
        # Initialize edge encoder system
        self.edge_encoder = get_edge_encoder()
        
        # Set provider based on model_name
        if model_name == "local_gpt":
            self.edge_encoder.primary_provider = LLMProvider.LOCAL_GPT
        elif model_name == "ollama":
            self.edge_encoder.primary_provider = LLMProvider.OLLAMA
        elif model_name == "horizon_beta":
            self.edge_encoder.primary_provider = LLMProvider.OPENROUTER_HORIZON
        elif model_name == "openai_gpt5_auto":
            self.edge_encoder.primary_provider = LLMProvider.OPENAI_GPT5_AUTO
        elif model_name == "openai_gpt5":
            self.edge_encoder.primary_provider = LLMProvider.OPENAI_GPT5
        elif model_name == "openai_gpt5_mini":
            self.edge_encoder.primary_provider = LLMProvider.OPENAI_GPT5_MINI
        elif model_name == "openai_gpt5_nano":
            self.edge_encoder.primary_provider = LLMProvider.OPENAI_GPT5_NANO
        else:  # Default to hybrid edge encoding
            self.edge_encoder.primary_provider = LLMProvider.HYBRID_EDGE
        
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
            # Initialize ChromaDB only if available and embedding_model exists
            if chromadb is not None and self.embedding_model is not None:
                self.vector_db = chromadb.Client(Settings(
                    persist_directory=".chromadb",
                    anonymized_telemetry=False
                ))
                self.collection = self.vector_db.get_or_create_collection(
                    name="hidden_words",
                    metadata={"hnsw:space": "cosine"}
                )
                if self.collection.count() == 0:
                    self._process_text()
            else:
                self.vector_db = None
                self.collection = None
        except Exception as e:
            console.print(f"[red]Error initializing vector database:[/red] {str(e)}")
    
    def _load_raw_verses(self, path: str) -> List[str]:
        """Load verses from text file for naive retrieval fallback."""
        try:
            with open(path, 'r') as f:
                text = f.read()
            verses = []
            for v in text.split('\n\n'):
                s = v.strip()
                # Skip headers/metadata and very short chunks
                if not s or s.startswith('#') or len(s) < 40:
                    continue
                verses.append(s)
            return verses
        except Exception:
            return []
    
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
            
            if self.embedding_model is not None and self.collection is not None:
                embeddings = self.embedding_model.encode(words)
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
            # Embedding-based retrieval if available
            if self.embedding_model is not None and getattr(self, 'collection', None) is not None:
                query_embedding = self.embedding_model.encode([query])[0]
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                return results.get('documents', [[]])[0]
            
            # Fallback: naive retrieval over raw verses
            if self.raw_verses:
                text = query.lower()
                # Derive simple keywords by splitting and removing short tokens
                tokens = [t for t in re.split(r"[^a-zA-Z']+", text) if len(t) >= 3]
                # If no tokens, just return first verses
                if not tokens:
                    return self.raw_verses[:top_k]
                # Score verses by keyword occurrence count
                scored: List[tuple[int, str]] = []
                for verse in self.raw_verses:
                    vlow = verse.lower()
                    score = sum(1 for t in tokens if t in vlow)
                    if score > 0:
                        scored.append((score, verse))
                if not scored:
                    return self.raw_verses[:top_k]
                scored.sort(key=lambda x: x[0], reverse=True)
                return [v for _, v in scored[:top_k]]
            
            return []
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
        text = message.lower()
        word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}

        # Patterns that indicate a quantity of quotes (avoid false matches like "one line")
        qty_patterns = [
            r"\b(give me|share|provide|send|show|list)\s+(one|two|three|four|five|[1-5])\s+(quote|quotes)\b",
            r"\b(one|two|three|four|five|[1-5])\s+(quote|quotes)\b",
        ]

        for pat in qty_patterns:
            m = re.search(pat, text)
            if m:
                token = m.group(2) if len(m.groups()) >= 2 else m.group(1)
                if token in word_to_num:
                    return word_to_num[token]
                try:
                    return int(token)
                except Exception:
                    return 1

        return 1
    
    def _call_horizon_beta(self, messages: List[Dict[str, str]]) -> str:
        """Call Horizon Beta via OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use proper OpenRouter Horizon Beta model name
        model_name = "openrouter/horizon-beta" if self.model_name == "hybrid_edge" else self.model_name
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                console.print(f"[red]Error calling Horizon Beta:[/red] {response.status_code} - {response.text}")
                return "I'm having trouble connecting right now. Can you try again?"
        except Exception as e:
            console.print(f"[red]Error calling Horizon Beta:[/red] {str(e)}")
            return "I'm having trouble connecting right now. Can you try again?"
    
    def _call_openai_gpt5(self, messages: List[Dict[str, str]], model: str = "gpt-5-mini") -> str:
        """Call OpenAI GPT-5 family directly if key available."""
        api_key = OPENAI_API_KEY
        if not api_key:
            return ""
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return ""
        except Exception:
            return ""

    def _call_ollama_text(self, prompt: str) -> str:
        """Best-effort local generation via Ollama if running."""
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json().get("response", "")
            return ""
        except Exception:
            return ""

    def chat(self, message: str) -> str:
        """Process a message and return the agent's response using edge encoding."""
        try:
            self.conversation_history.append({"role": "user", "content": message})
            is_emotional = self._update_user_context(message)

            # If no OpenRouter API key, skip cloud generation and use local fallback
            if not OPENROUTER_API_KEY:
                response = self._fallback_response(message)
            else:
                context = self._get_conversation_context()
                # Run async call safely whether called from sync or async context
                try:
                    try:
                        running = asyncio.get_running_loop()
                        # Already inside an event loop: dispatch to thread
                        encoding_result = asyncio.run(asyncio.to_thread(
                            lambda: asyncio.run(self.edge_encoder.encode_query(message, context))
                        ))
                    except RuntimeError:
                        # No running loop: create a fresh one
                        response_future = self.edge_encoder.encode_query(message, context)
                        encoding_result = asyncio.run(response_future)
                    response = encoding_result.get("final_response", "")
                except Exception as _err:
                    response = ""
                if not response:
                    response = self._fallback_response(message)
            # Ensure quotes only for explicit spiritual/quote requests (deterministic, low cost)
            quote_triggers = [
                "quote", "hidden words", "spiritual guidance", 
                "what does it say", "share wisdom", "teachings",
                "quotation", "from the hidden words", "spiritual quote",
                "any quote", "any quotation", "share a quote",
                "retrieve", "get", "find", "show me"
            ]
            needs_quotes = any(t in message.lower() for t in quote_triggers)
            if needs_quotes:
                count = self._extract_quote_count(message)
                verses = self._retrieve_relevant_words(message, top_k=count)
                cleaned = [self._clean_quote(v).strip() for v in verses if v and v.strip()]
                if cleaned:
                    # For minimal cost and deterministic formatting, return pre-styled card HTML
                    cards = []
                    for q in cleaned:
                        cards.append(
                            '<div class="hidden-word-quote"><div class="quote-content">' +
                            f'<div class="quote-text">{q}</div>' +
                            '</div></div>'
                        )
                    response = "\n\n".join(cards)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            console.print(f"[red]Error in chat:[/red] {str(e)}")
            return self._fallback_response(message)
    
    def _get_conversation_context(self) -> str:
        """Get conversation context for edge encoding"""
        if len(self.conversation_history) > 1:
            # Get last few messages for context
            recent_messages = self.conversation_history[-4:]  # Last 4 messages
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
            return context
        return ""
    
    def _fallback_response(self, message: str) -> str:
        """Fallback response when edge encoding fails"""
        # Check for explicit requests for quotes
        quote_triggers = [
            "quote", "hidden words", "spiritual guidance", 
            "what does it say", "share wisdom", "teachings",
            "quotation", "from the hidden words", "spiritual quote",
            "any quote", "any quotation", "share a quote",
            "retrieve", "get", "find", "show me"
        ]
        
        # Check for emotional state
        is_emotional = self._update_user_context(message)
        
        has_triggers = any(trigger in message.lower() for trigger in quote_triggers)
        # If there is no explicit quote trigger, generate with external model chain (Horizon -> OpenAI -> Ollama),
        # and fall back to local empathy only if all fail. No quotes allowed in this branch.
        if not has_triggers:
            system = {
                "role": "system",
                "content": "You are a normal, friendly person having a casual conversation. Be concise, kind, and do NOT include any spiritual content or quotes.",
            }
            user = {"role": "user", "content": message}
            response = ""
            # 1) Horizon if key
            if OPENROUTER_API_KEY:
                response = self._call_horizon_beta([system, user])
            # 2) OpenAI GPT‑5 Mini if key
            if not response and OPENAI_API_KEY:
                response = self._call_openai_gpt5([system, user], model="gpt-5-mini")
            # 3) Ollama local if available
            if not response:
                response = self._call_ollama_text(f"System: {system['content']}\nUser: {message}\nAssistant:")
            # 4) Final fallback: local empathy
            agent_response = self._clean_response(response) if response else (
                "I'm here with you. That sounds tough—would you like to share a bit more about how you're feeling?" if is_emotional else
                "I'm here with you. Tell me more—how's your day going?"
            )
            # Avoid UI quote-card detection for normal talk by stripping straight quotes
            agent_response = agent_response.replace('"', '')
            
        else:
            # Explicit quotes: build pre-styled quote cards from retrieved verses (deterministic)
            quote_count = self._extract_quote_count(message)
            relevant_words = self._retrieve_relevant_words(message, top_k=quote_count)
            cleaned = [self._clean_quote(q).strip() for q in relevant_words if q and q.strip()]
            if cleaned:
                cards = []
                for q in cleaned[:max(1, quote_count)]:
                    cards.append(
                        '<div class="hidden-word-quote"><div class="quote-content">'
                        f'<div class="quote-text">{q}</div>'
                        '</div></div>'
                    )
                agent_response = "\n\n".join(cards)
            else:
                # If no local verses, try a model to generate quotes
                messages = [
                    {"role": "system", "content": "Share concise Hidden Words quote(s) only, no verse numbers."},
                    {"role": "user", "content": message},
                ]
                response = self._call_horizon_beta(messages) if OPENROUTER_API_KEY else ""
                if not response and OPENAI_API_KEY:
                    response = self._call_openai_gpt5(messages, model="gpt-5-mini")
                agent_response = self._clean_quote(response) if response else "Please try again in a moment."
                # Ensure at least one explicit quoted passage is present for UI formatting
                if '"' not in agent_response and relevant_words:
                    first = self._clean_quote(relevant_words[0]).strip()
                    if first:
                        agent_response = f'"{first}"\n\n' + agent_response
        
        # Add agent response to history
        self.conversation_history.append({"role": "assistant", "content": agent_response})
        
        return agent_response
    
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
