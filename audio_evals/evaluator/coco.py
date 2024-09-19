from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.coco import compute_caption


class Coco(Evaluator):

    def _eval(self, pred: str, label: str, **kwargs):
        pred, label = str(pred), str(label)
        return compute_caption([label], [pred])
