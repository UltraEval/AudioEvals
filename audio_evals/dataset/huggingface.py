import logging
import os
from typing import Dict, List, Optional

import soundfile as sf
from datasets import load_dataset

from audio_evals.dataset.dataset import Dataset as BaseDataset

logger = logging.getLogger(__name__)


def load_audio_hf_dataset(name, cfg=None, split=""):
    if split:
        ds = load_dataset(name, name=cfg, split=split)

        save_path = f"raw/{name}/{split}/"
        if cfg:
            save_path = f"raw/{name}/{cfg}/{split}/"
        os.makedirs(save_path, exist_ok=True)

        def save_audio(example):
            audio_array = example["audio"]["array"]
            output_path = os.path.join(save_path, example["audio"]["path"])
            example["WavPath"] = output_path
            d = os.path.dirname(output_path)
            os.makedirs(d, exist_ok=True)
            if not os.path.exists(output_path):
                sf.write(output_path, audio_array, example["audio"]["sampling_rate"])
            return example

        ds = ds.map(save_audio)
        return list(ds)
    else:
        dses = load_dataset(name)
        result = []
        for k in dses:
            result.extend(load_audio_hf_dataset(name, k))
        return result


class Huggingface(BaseDataset):
    def __init__(
        self,
        name: str,
        default_task: str,
        ref_col: str,
        cfg: Optional[str] = None,
        split: str = "",
    ):
        super().__init__(default_task, ref_col)
        self.name = name
        self.cfg = cfg
        self.split = split

    def load(self) -> List[Dict[str, any]]:
        logger.info(
            "start load data, it will take a while for download dataset when first load dataset"
        )
        return load_audio_hf_dataset(self.name, self.cfg, self.split)
