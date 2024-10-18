import os
from typing import Dict, Any
import openai

from audio_evals.models.model import APIModel
from audio_evals.base import PromptStruct


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com")


class GPT(APIModel):
    def __init__(self, model_name: str, sample_params: Dict[str, Any] = None):
        super().__init__(True, sample_params)
        self.model_name = model_name
        assert "OPENAI_API_KEY" in os.environ, ValueError(
            "not found OPENAI_API_KEY in your ENV"
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        messages = []
        for item in prompt:
            messages.append(
                {"role": item["role"], "content": item["contents"][0]["value"]}
            )

        response = openai.ChatCompletion.create(
            model=self.model_name, messages=messages, **kwargs
        )

        return response.choices[0].message["content"]
