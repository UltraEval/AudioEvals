import pytest

from audio_evals.process.qwen import QwenAudioASRExtract


def test_qwen_audio_asr_extract():
    ex = QwenAudioASRExtract('zh')
    for s in ['''买一张万能卡也有不少好处带着这张卡你可以进入南非的一些公园或全部的国家公园<|endoftext|>''',
              ]:
        print(ex(s))
