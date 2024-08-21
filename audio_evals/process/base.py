import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Process(ABC):
    @abstractmethod
    def __call__(self, answer: str) -> str:
        raise NotImplementedError()


class ContentExtract(Process):

    def __call__(self, answer: str) -> str:
        try:
            return json.loads(answer)["content"]
        except Exception as e:
            logger.warning(f"process {answer} fail: {str(e)}")
        return answer
