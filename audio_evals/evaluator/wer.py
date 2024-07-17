from jiwer import wer, cer

from audio_evals.evaluator.base import Evaluator


class WER(Evaluator):

    def __call__(self, pred, label, **kwargs):
        return {'match': wer(label, pred)}


class CER(Evaluator):

    def __call__(self, pred, label, **kwargs):
        return {'match': cer(label, pred)}

