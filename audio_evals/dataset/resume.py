import json
from typing import List, Dict


class ResumeDataset:
    def __init__(self, raw_dataset: str, resume_file: str):
        from audio_evals.registry import registry

        raw_dataset = registry.get_dataset(raw_dataset)
        self.raw_dataset = raw_dataset
        self.task_name = raw_dataset.task_name
        self.ref_col = raw_dataset.ref_col
        self.resume_file = resume_file

    def load(self) -> List[Dict[str, any]]:
        data = self.raw_dataset.load()
        with open(self.resume_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                if doc["type"] == "error":
                    continue
                idx = int(doc["id"])
                data[idx]["eval_info"] = (
                    data[idx].get("eval_info", {}).update({doc["type"]: doc["data"]})
                )
        return data
