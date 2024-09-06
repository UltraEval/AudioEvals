import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.utils import retry

# the str type for pre-train model, the list type for chat model

logger = logging.getLogger(__name__)


class Model(ABC):
    def __init__(self, is_chat: bool, sample_params: Dict[str, any] = None):
        self.is_chat = is_chat
        if sample_params is None:
            sample_params = {}
        self.sample_params = sample_params

    @abstractmethod
    def _inference(self, prompt: PromptStruct, **kwargs):
        raise NotImplementedError()

    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        if isinstance(prompt, list) and not self.is_chat:
            raise ValueError("struct input not match pre-train model")
        if isinstance(prompt, str) and self.is_chat:
            prompt = [{"role": "user", "contents": [{"type": "text", "value": prompt}]}]
        sample_params = deepcopy(self.sample_params)
        sample_params.update(kwargs)
        logger.debug(f"sample_params: {sample_params}\nprompt: {prompt}")
        return self._inference(prompt, **sample_params)


class APIModel(Model, ABC):

    @retry(max_retries=3)
    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        return super().inference(prompt, **kwargs)
