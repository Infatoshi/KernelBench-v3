"""
Unified LLM Model Wrapper supporting multiple providers:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude)
- xAI (Grok)
- Google (Gemini)
- Ollama (local)
- vLLM (local server)
- SGLang (local server)
- Groq
- Cerebras

All providers use a common interface for easy switching.
"""

import os
import json
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential

from llm_apis.models.Base import BaseModel


class UnifiedModel(BaseModel):
    """Unified model wrapper supporting multiple LLM providers."""
    
    PROVIDER_CONFIGS = {
        'openai': {
            'base_url': 'https://api.openai.com/v1',
            'api_key_env': 'OPENAI_API_KEY',
            'default_model': 'gpt-4o',
            'type': 'openai'
        },
        'anthropic': {
            'base_url': 'https://api.anthropic.com',
            'api_key_env': 'ANTHROPIC_API_KEY',
            'default_model': 'claude-3-5-sonnet-20241022',
            'type': 'anthropic'
        },
        'xai': {
            'base_url': 'https://api.x.ai/v1',
            'api_key_env': 'XAI_API_KEY',
            'default_model': 'grok-2-latest',
            'type': 'openai'
        },
        'gemini': {
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
            'api_key_env': 'GEMINI_API_KEY',
            'default_model': 'gemini-2.0-flash-exp',
            'type': 'gemini'
        },
        'groq': {
            'base_url': 'https://api.groq.com/openai/v1',
            'api_key_env': 'GROQ_API_KEY',
            'default_model': 'llama-3.3-70b-versatile',
            'type': 'openai'
        },
        'cerebras': {
            'base_url': 'https://api.cerebras.ai/v1',
            'api_key_env': 'CEREBRAS_API_KEY',
            'default_model': 'llama-3.3-70b',
            'type': 'cerebras'
        },
        'ollama': {
            'base_url': 'http://localhost:11434/v1',
            'api_key_env': None,  # Ollama doesn't need API key
            'default_model': 'llama3.1',
            'type': 'openai'
        },
        'vllm': {
            'base_url': 'http://localhost:8000/v1',
            'api_key_env': None,  # Local server
            'default_model': 'meta-llama/Llama-3.1-8B-Instruct',
            'type': 'openai'
        },
        'sglang': {
            'base_url': 'http://localhost:30000/v1',
            'api_key_env': None,  # Local server
            'default_model': 'default',
            'type': 'openai'
        }
    }
    
    def __init__(
        self,
        provider: str = 'openai',
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize unified model client.
        
        Args:
            provider: Provider name (openai, anthropic, xai, gemini, groq, cerebras, ollama, vllm, sglang)
            model_id: Model identifier (uses provider default if not specified)
            api_key: API key (uses env var if not specified)
            base_url: Custom base URL (uses provider default if not specified)
            **kwargs: Additional provider-specific arguments
        """
        provider = provider.lower()
        
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {', '.join(self.PROVIDER_CONFIGS.keys())}"
            )
        
        config = self.PROVIDER_CONFIGS[provider]
        self.provider = provider
        self.provider_type = config['type']
        
        # Set model ID
        self.model_id = model_id or config['default_model']
        
        # Set base URL
        self.base_url = base_url or config['base_url']
        
        # Set API key
        if config['api_key_env']:
            self.api_key = api_key or os.getenv(config['api_key_env'])
            if not self.api_key:
                raise ValueError(
                    f"API key not provided. Set {config['api_key_env']} "
                    f"environment variable or pass api_key parameter."
                )
        else:
            # Local servers don't need API keys
            self.api_key = api_key or "not-needed"
        
        # Initialize the appropriate client
        self._init_client(**kwargs)
    
    def _init_client(self, **kwargs):
        """Initialize provider-specific client."""
        if self.provider_type == 'openai':
            # OpenAI-compatible providers
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs
            )
            
        elif self.provider_type == 'anthropic':
            # Anthropic Claude
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=self.api_key,
                **kwargs
            )
            
        elif self.provider_type == 'gemini':
            # Google Gemini
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_id)
            
        elif self.provider_type == 'cerebras':
            # Cerebras can use their SDK or OpenAI-compatible endpoint
            try:
                from cerebras.cloud.sdk import Cerebras
                self.client = Cerebras(
                    api_key=self.api_key,
                    **kwargs
                )
            except ImportError:
                # Fallback to OpenAI-compatible
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs
                )
                self.provider_type = 'openai'  # Use OpenAI interface
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(
        self,
        messages: List,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        **kwargs
    ) -> str:
        """
        Generate text from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        if self.provider_type == 'openai':
            return self._generate_openai(messages, temperature, max_tokens, **kwargs)
        elif self.provider_type == 'anthropic':
            return self._generate_anthropic(messages, temperature, max_tokens, **kwargs)
        elif self.provider_type == 'gemini':
            return self._generate_gemini(messages, temperature, max_tokens, **kwargs)
        elif self.provider_type == 'cerebras':
            return self._generate_cerebras(messages, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown provider type: {self.provider_type}")
    
    def _generate_openai(
        self,
        messages: List,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        messages: List,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Anthropic Claude API."""
        # Anthropic uses different format - extract system message
        system_msg = None
        formatted_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                formatted_messages.append(msg)
        
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg or "",
            messages=formatted_messages,
            **kwargs
        )
        return response.content[0].text
    
    def _generate_gemini(
        self,
        messages: List,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Google Gemini API."""
        import google.generativeai as genai
        
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = 'user' if msg['role'] in ['user', 'system'] else 'model'
            gemini_messages.append({
                'role': role,
                'parts': [msg['content']]
            })
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )
        
        # Generate
        response = self.client.generate_content(
            gemini_messages,
            generation_config=generation_config
        )
        return response.text
    
    def _generate_cerebras(
        self,
        messages: List,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using Cerebras API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback: treat as OpenAI-compatible
            return self._generate_openai(messages, temperature, max_tokens, **kwargs)
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the provider with a simple request.
        
        Returns:
            Dict with status and response information
        """
        test_messages = [{"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}]
        
        try:
            response = self.generate(
                messages=test_messages,
                temperature=0,
                max_tokens=50
            )
            return {
                'status': 'success',
                'provider': self.provider,
                'model': self.model_id,
                'response': response,
                'error': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'provider': self.provider,
                'model': self.model_id,
                'response': None,
                'error': str(e)
            }
    
    def __repr__(self):
        return f"UnifiedModel(provider={self.provider}, model={self.model_id})"

