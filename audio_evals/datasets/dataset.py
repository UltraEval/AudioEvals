import json
import os.path
from abc import ABC, abstractmethod
from typing import List, Dict, Generator


class Dataset(ABC):
    def __init__(self, f_name: str, default_task: str, ref_col: str):
        self.f_name = f_name
        self.task_name = default_task
        self.ref_col = ref_col

    @abstractmethod
    def load(self) -> Generator[Dict[str, any], None, None]:
        raise NotImplementedError()


class ASR(Dataset):

    def load(self) -> Generator[Dict[str, any], None, None]:
        with open(self.f_name) as f:
            for line in f:
                yield json.loads(line)


class RelativeASR(Dataset):
    def __init__(self, f_name: str, default_task: str, ref_col: str, file_path_prefix: str):
        super().__init__(f_name, default_task, ref_col)
        if not file_path_prefix.endswith('/'):
            file_path_prefix += '/'
        self.file_path = file_path_prefix

    def load(self) -> Generator[Dict[str, any], None, None]:
        with open(self.f_name) as f:
            for line in f:
                doc = json.loads(line)
                for k, v in doc.items():
                    temp = os.path.join(self.file_path, v)
                    if os.path.exists(temp):
                        doc[k] = temp
                yield doc
