import logging
import re
from typing import Dict

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

logger = logging.getLogger(__name__)


class WhisperModel(Model):
    def __init__(
        self,
        path: str,
        sample_params: Dict[str, any] = None,
    ):
        super().__init__(True, sample_params)  # as a chat model
        logger.debug("start load model from {}".format(path))

        # Load the speech recognition model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).eval()
        self.model.to(self.device)

        logger.debug("successfully load model from {}".format(path))

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(path)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:

        audio = ""

        for content in prompt:
            if content["role"] == "user":
                for line in content["contents"]:
                    if line["type"] == "audio":
                        audio = line["value"]
                        break
                    elif line["type"] == "text":
                        m = re.match(
                            r"translate this audio to (.*?),",
                            line["value"],
                            re.IGNORECASE,
                        )
                        if m:
                            kwargs["generate_kwargs"] = {
                                "language": m.group(1),
                                "task": "translate",
                            }

        logger.info(f"the input is {audio}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        result = pipe(audio, **kwargs)
        return result["text"]
