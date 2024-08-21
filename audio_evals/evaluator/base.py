from abc import ABC, abstractmethod
from typing import Dict


class Evaluator(ABC):

    @abstractmethod
    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        raise NotImplementedError()

    def __call__(self, pred, label, **kwargs) -> Dict[str, any]:
        res = {"pred": pred, "ref": label}
        res.update(self._eval(pred, label, **kwargs))
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

        return {"match": 1 if pred == label else 0, "pred": pred, "label": label}


class PrefixMatch(Evaluator):

    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        if self.ignore_case:
            pred = pred.lower().strip()
            label = str(label).lower().strip()
        n = len(label)
        return {
            "match": 1 if pred[:n] == label else 0,
            "pred": pred[:n],
            "label": label,
        }
