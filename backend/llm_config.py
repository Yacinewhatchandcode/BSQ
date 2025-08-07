#!/usr/bin/env python3
"""
LLM Configuration System - Edge Encoders with Local GPT and OpenRouter Horizon Beta
Provides configurable AI model selection and routing
"""

import os
import json
import requests
import subprocess
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    LOCAL_GPT = "local_gpt"
    OLLAMA = "ollama"
    OPENROUTER_HORIZON = "openrouter_horizon"
    HYBRID_EDGE = "hybrid_edge"  # Uses both local and cloud for edge encoding

class LLMConfig:
    """Configuration for different LLM providers"""
    
    def __init__(self):
        self.providers = {
            LLMProvider.LOCAL_GPT: {
                "name": "Local GPT",
                "endpoint": "http://localhost:8080/v1/chat/completions",
                "model": "gpt-3.5-turbo",
                "available": self._check_local_gpt(),
                "description": "Local GPT instance running on your machine"
            },
            LLMProvider.OLLAMA: {
                "name": "Ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model": "qwen2.5:7b",
                "available": self._check_ollama(),
                "description": "Local Ollama with Qwen model"
            },
            LLMProvider.OPENROUTER_HORIZON: {
                "name": "OpenRouter Horizon Beta",
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "model": "openrouter/horizon-beta",
                "api_key": "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",
                "available": True,
                "description": "Cutting-edge Horizon Beta via OpenRouter"
            },
            LLMProvider.HYBRID_EDGE: {
                "name": "Hybrid Edge Encoder",
                "description": "Uses local LLM for encoding, Horizon Beta for generation",
                "available": True
            }
        }
    
    def _check_local_gpt(self) -> bool:
        """Check if local GPT is running"""
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available LLM providers"""
        return [
            {
                "id": provider.value,
                "name": config["name"],
                "description": config["description"],
                "available": config["available"]
            }
            for provider, config in self.providers.items()
        ]

class EdgeEncoder:
    """Edge encoding system using multiple LLMs for optimal performance"""
    
    def __init__(self, primary_provider: LLMProvider = LLMProvider.HYBRID_EDGE):
        self.config = LLMConfig()
        self.primary_provider = primary_provider
        self.fallback_provider = LLMProvider.OPENROUTER_HORIZON
    
    async def encode_query(self, query: str, context: str = "") -> Dict[str, Any]:
        """Encode query using edge LLM for preprocessing"""
        if self.primary_provider == LLMProvider.HYBRID_EDGE:
            return await self._hybrid_encode(query, context)
        else:
            return await self._single_provider_encode(query, context, self.primary_provider)
    
    async def _hybrid_encode(self, query: str, context: str) -> Dict[str, Any]:
        """Use local LLM for encoding, Horizon Beta for generation"""
        # Step 1: Use local LLM for query analysis and encoding
        local_analysis = await self._analyze_locally(query, context)
        
        # Step 2: Use Horizon Beta for final generation with encoded context
        final_response = await self._generate_with_horizon(query, local_analysis)
        
        return {
            "query": query,
            "local_analysis": local_analysis,
            "final_response": final_response,
            "encoding_method": "hybrid_edge",
            "providers_used": ["local", "horizon_beta"]
        }
    
    async def _analyze_locally(self, query: str, context: str) -> Dict[str, Any]:
        """Analyze query using local LLM"""
        # Try Ollama first, then local GPT
        for provider in [LLMProvider.OLLAMA, LLMProvider.LOCAL_GPT]:
            if self.config.providers[provider]["available"]:
                try:
                    return await self._call_local_llm(query, context, provider)
                except Exception as e:
                    logger.warning(f"Local LLM {provider.value} failed: {e}")
                    continue
        
        # Fallback to basic analysis
        return {
            "intent": "spiritual_guidance",
            "keywords": query.split(),
            "context_summary": context[:200] if context else "",
            "encoding": "basic_fallback"
        }
    
    async def _call_local_llm(self, query: str, context: str, provider: LLMProvider) -> Dict[str, Any]:
        """Call local LLM for query analysis"""
        config = self.config.providers[provider]
        
        analysis_prompt = f"""Analyze this spiritual query for The Hidden Words knowledge base:
Query: {query}
Context: {context}

Provide analysis in JSON format:
- intent: primary intention (spiritual_guidance, question, reflection, etc.)
- keywords: key terms for search
- emotional_tone: detected emotional state
- spiritual_themes: relevant spiritual concepts
- search_strategy: how to search the knowledge base
"""
        
        if provider == LLMProvider.OLLAMA:
            return await self._call_ollama(analysis_prompt)
        elif provider == LLMProvider.LOCAL_GPT:
            return await self._call_local_gpt(analysis_prompt)
    
    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=10
            )
            result = response.json()
            
            # Try to parse JSON from response
            try:
                return json.loads(result.get("response", "{}"))
            except:
                return {
                    "intent": "spiritual_guidance",
                    "analysis": result.get("response", ""),
                    "provider": "ollama"
                }
        except Exception as e:
            raise Exception(f"Ollama call failed: {e}")
    
    async def _call_local_gpt(self, prompt: str) -> Dict[str, Any]:
        """Call local GPT API"""
        try:
            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=10
            )
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Try to parse JSON from response
            try:
                return json.loads(content)
            except:
                return {
                    "intent": "spiritual_guidance",
                    "analysis": content,
                    "provider": "local_gpt"
                }
        except Exception as e:
            raise Exception(f"Local GPT call failed: {e}")
    
    async def _generate_with_horizon(self, query: str, analysis: Dict[str, Any]) -> str:
        """Generate final response using Horizon Beta with local analysis"""
        config = self.config.providers[LLMProvider.OPENROUTER_HORIZON]
        
        enhanced_prompt = f"""You are a spiritual guide for The Hidden Words by Bahá'u'lláh.

Original Query: {query}

Local Analysis:
- Intent: {analysis.get('intent', 'spiritual_guidance')}
- Keywords: {analysis.get('keywords', [])}
- Emotional Tone: {analysis.get('emotional_tone', 'seeking')}
- Spiritual Themes: {analysis.get('spiritual_themes', [])}

Using this analysis, provide a thoughtful, spiritually enriching response that draws from The Hidden Words and Baha'i teachings. Be warm, wise, and encouraging.
"""
        
        try:
            response = requests.post(
                config["endpoint"],
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config["model"],
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "temperature": 0.7
                },
                timeout=30
            )
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Horizon Beta call failed: {e}")
            return f"I understand you're asking about {query}. While I'm having trouble accessing my full knowledge right now, I can share that The Hidden Words teaches us about spiritual growth and divine love. Please try your question again."
    
    async def _single_provider_encode(self, query: str, context: str, provider: LLMProvider) -> Dict[str, Any]:
        """Use single provider for encoding and generation"""
        config = self.config.providers[provider]
        
        if provider == LLMProvider.OPENROUTER_HORIZON:
            response = await self._generate_with_horizon(query, {"intent": "spiritual_guidance"})
            return {
                "query": query,
                "final_response": response,
                "encoding_method": "single_provider",
                "provider_used": provider.value
            }
        else:
            # For local providers, use them for both analysis and generation
            analysis = await self._analyze_locally(query, context)
            return {
                "query": query,
                "local_analysis": analysis,
                "final_response": analysis.get("analysis", "I'm here to help with your spiritual journey."),
                "encoding_method": "single_provider",
                "provider_used": provider.value
            }

# Global edge encoder instance
edge_encoder = EdgeEncoder()

def get_llm_config() -> LLMConfig:
    """Get LLM configuration instance"""
    return LLMConfig()

def get_edge_encoder() -> EdgeEncoder:
    """Get edge encoder instance"""
    return edge_encoder

def set_primary_provider(provider: LLMProvider):
    """Set primary LLM provider"""
    global edge_encoder
    edge_encoder.primary_provider = provider
    logger.info(f"Primary LLM provider set to: {provider.value}")
