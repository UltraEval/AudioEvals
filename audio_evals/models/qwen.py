import logging
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


def process_prompts(prompt: PromptStruct):
    def _conv_contents(content):
        content = content.deepcopy()
        for ele in content:
            if ele["type"] == "audio":
                ele["audio_url"] = ele["value"]
            del ele["value"]
        return content

    for line in prompt:
        cont = []
        for cont_ele in line["contents"]:
            cont.append(_conv_contents(cont_ele))
        line["content"] = cont
        del line["contents"]

    return prompt


class Qwen2audio(Model):
    def __init__(self, path: str, sample_params: Dict[str, any] = None):
        super().__init__(True, sample_params)
        logger.debug("start load model from {}".format(path))
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            path,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        logger.debug("successfully load model from {}".format(path))
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    def _inference(self, prompt: PromptStruct, **kwargs):
        prompt = process_prompts(prompt)
        text = self.processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in prompt:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(
                            ele["audio_url"],
                            sr=self.processor.feature_extractor.sampling_rate)[0])

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.input_ids.to("cuda")

        generate_ids = self.model.generate(**inputs, **kwargs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response
