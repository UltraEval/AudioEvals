from abc import ABC, abstractmethod
from typing import Dict, List


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


class Dump(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        return {}


class ACC(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        if len(score_detail) == 0:
            return {'acc': 0}
        assert 'match' in score_detail[0]
        return {'acc': sum([item['match'] for item in score_detail]) / len(score_detail)}


class Mean(ACC):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        res = super()._agg(score_detail)
        return {'mean': res['acc']}
