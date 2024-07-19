from abc import ABC, abstractmethod
from typing import Dict


class Evaluator(ABC):
    @abstractmethod
    def __call__(self, pred, label, **kwargs) -> Dict[str, any]:
        raise NotImplementedError()


class Dump(Evaluator):

    def __call__(self, *args, **kwargs):
        return {}


class EM(Evaluator):

    def __call__(self, pred, label, **kwargs) -> Dict[str, any]:
        if type(label) in [int, float]:
            pred, label = float(pred), float(label)
        elif isinstance(label, str):
            pred, label = str(pred).strip(), label.strip()

        return {'match': 1 if pred == label else 0, 'pred': pred, 'label': label}
