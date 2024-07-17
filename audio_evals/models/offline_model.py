from audio_evals.base import PromptStruct
from audio_evals.models.model import Model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import logging


logger = logging.getLogger(__name__)


class OfflineModel(Model):
    def __init__(self, is_chat: bool, path: str):
        super().__init__(is_chat)
        logger.debug('start load model from {}'.format(path))
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        logger.debug('successfully load model from {}'.format(path))

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    def _inference(self, prompt: PromptStruct, **kwargs):
        vl_list = [{"audio": s["value"]} if s["type"] == "audio" else {"text": s["value"]} for line in prompt for s in line["contents"]]
        logger.debug(f"the input is {vl_list}")
        query = self.tokenizer.from_list_format(vl_list)
        generated_text, _ = self.model.chat(self.tokenizer, query=query, history=None, **kwargs)
        logger.debug(f"the output is {generated_text}")
        return generated_text
