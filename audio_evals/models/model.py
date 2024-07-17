import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Union, List, Dict

import aiohttp

from audio_evals.base import PromptStruct
from audio_evals.utils import retry

# the str type for pre-train model, the list type for chat model


class Model(ABC):
    def __init__(self, is_chat: bool):
        self.is_chat = is_chat

    @abstractmethod
    def _inference(self, prompt: PromptStruct, **kwargs):
        raise NotImplementedError()

    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        if isinstance(prompt, list) and not self.is_chat:
            raise ValueError('struct input not match pre-train model')

        return self._inference(prompt, **kwargs)


class APIModel(Model):

    @retry(max_retries=3, default='')
    @abstractmethod
    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        raise NotImplementedError()

