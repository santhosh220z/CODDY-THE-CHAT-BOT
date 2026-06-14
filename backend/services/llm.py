import json
from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any, Optional
import requests
import httpx
from backend.utils.config import settings
from backend.utils.logging import logger

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a complete text response."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Stream the generated response token by token."""
        pass

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip('/')
        self.model = settings.OLLAMA_MODEL
        logger.info(f"Initialized Ollama LLM provider (base_url: {self.base_url}, model: {self.model})")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: Unable to connect to local Ollama server at {self.base_url}. Make sure Ollama is running and the model '{self.model}' is installed."

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            with requests.post(url, json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        token = chunk.get("response", "")
                        yield token
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"\n[Error: Unable to connect to local Ollama server. Check your connection or logs. Details: {str(e)}]"


class HuggingFaceProvider(LLMProvider):
    def __init__(self):
        self.api_key = settings.HF_API_KEY
        self.model = settings.HF_MODEL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        logger.info(f"Initialized Hugging Face provider (model: {self.model})")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: Hugging Face API key is missing. Set HF_API_KEY in your .env file."
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:" if system_prompt else prompt
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "return_full_text": False
            }
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            res = response.json()
            if isinstance(res, list) and len(res) > 0:
                return res[0].get("generated_text", "")
            return str(res)
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            return f"Error: Hugging Face API call failed. Details: {e}"

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        if not self.api_key:
            yield "Error: Hugging Face API key is missing. Set HF_API_KEY in your .env file."
            return
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:" if system_prompt else prompt
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "return_full_text": False
            },
            "stream": True
        }
        try:
            # Serverless streaming yields Server-Sent Events (SSE)
            with requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith("data:"):
                            data = json.loads(line_str[5:])
                            token = data.get("token", {}).get("text", "")
                            yield token
        except Exception as e:
            logger.error(f"Hugging Face streaming failed: {e}")
            yield f"\n[Error: Hugging Face streaming failed. Details: {e}]"


# Factory
_llm_instance = None

def get_llm_provider() -> LLMProvider:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
        
    provider_type = settings.LLM_PROVIDER.lower()
    if provider_type == "ollama":
        _llm_instance = OllamaProvider()
    elif provider_type == "huggingface":
        _llm_instance = HuggingFaceProvider()
    else:
        logger.warning(f"Unknown LLM provider: '{provider_type}'. Defaulting to Ollama.")
        _llm_instance = OllamaProvider()
        
    return _llm_instance
