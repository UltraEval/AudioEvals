from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import sacrebleu
from jiwer import wer, cer


class AggPolicy(ABC):
    def __init__(self, need_score_col: List[str] = None):
        self.need_score_col = need_score_col
        if need_score_col is None:
            self.need_score_col = []

    @abstractmethod
    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        raise NotImplementedError()

    def __call__(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        if len(score_detail) > 0:
            for col in self.need_score_col:
                assert col in score_detail[0], ValueError(f'not found {col} score')
        return self._agg(score_detail)


class WER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, ignore_case: bool = False):
        super().__init__(need_score_col)
        self.ignore_case = ignore_case

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item['pred']) for item in score_detail], [str(item['ref']) for item in score_detail]
        if self.ignore_case:
            predl, refl = [item.lower() for item in predl], [item.lower() for item in refl]
        return {'wer': wer(predl, refl)}


class CER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, ignore_case: bool = False):
        super().__init__(need_score_col)
        self.ignore_case = ignore_case

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item['pred']) for item in score_detail], [str(item['ref']) for item in score_detail]
        if self.ignore_case:
            predl, refl = [item.lower() for item in predl], [item.lower() for item in refl]
        return {'cer': cer(predl, refl)}


class Dump(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        return {}


class ACC(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        if len(score_detail) == 0:
            return {'acc': 0}
        df = pd.DataFrame(score_detail)
        res = {}
        for item in df.columns:
            try:
                res[item] = df[item].mean()
            except Exception:
                pass
        return res


class BLEU(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, lang: str = '13a'):
        super().__init__(need_score_col)
        self.lang = '13a'
        if lang == 'zh':
            self.lang = 'zh'
        elif lang == "ja":
            self.lang = "ja-mecab"

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item['pred']) for item in score_detail], [str(item) for item in score_detail]

        pred, ref = [], []
        for p, r in zip(predl, refl):
            if r:
                pred.append(p)
                ref.append(([r]))
        res = sacrebleu.corpus_bleu(pred, ref, tokenize=self.lang)
        return {'bleu': res.score}
