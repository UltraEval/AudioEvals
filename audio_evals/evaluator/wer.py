from jiwer import cer, wer

from audio_evals.evaluator.base import Evaluator


class WER(Evaluator):
    def __init__(self, ignore_case: bool = False):
        self.ignore_case = ignore_case

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        if self.ignore_case:
            pred, label = pred.lower(), label.lower()
        return {"match": wer(label, pred)}


class CER(Evaluator):
    def __init__(self, ignore_case: bool = False):
        self.ignore_case = ignore_case

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        if self.ignore_case:
            pred, label = pred.lower(), label.lower()
        return {"match": cer(label, pred)}
