from abc import ABC, abstractmethod
from typing import Dict


class Evaluator(ABC):

    @abstractmethod
    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        raise NotImplementedError()

    def __call__(self, pred, label, **kwargs) -> Dict[str, any]:
        res = self._eval(pred, label, **kwargs)
        res.update({'pred': pred, 'ref': label})
        return res


class Dump(Evaluator):

    def _eval(self, pred, label, **kwargs):
        return {}


class EM(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        if type(label) in [int, float]:
            pred, label = float(pred), float(label)
        elif isinstance(label, str):
            pred, label = str(pred).strip(), label.strip()

        return {'match': 1 if pred == label else 0, 'pred': pred, 'label': label}
