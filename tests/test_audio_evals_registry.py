import pytest

from audio_evals.eval_task import EvalTask
from audio_evals.recorder import Recorder
from audio_evals.registry import registry
import logging


# 配置根日志记录器
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])


def test_registry_model():
    model = registry.get_model('gpt4o')
    print(model.inference('how are you'))


def test_prompt():
    prompt = registry.get_prompt('asr')
    model = registry.get_model('qwen-audio')
    real_prompt = prompt.load(file='/Users/a1/Downloads/语音转文字/嘉德罗斯/嘉德罗斯_12.wav')
    print(model.inference(real_prompt))


def test_evaluator():
    e = registry.get_evaluator('em')
    assert e('0', 0)['match']
    assert e(0, '0')['match']
    assert e(1, '0')['match'] == 0


def test_agg():
    a = registry.get_agg('acc')
    assert a([{'match': 0}])['acc'] == 0
    assert a([{'match': 1}])['acc'] == 1
    assert a([])['acc'] == 0
    with pytest.raises(Exception):
        a([{'count': 1}])


def test_task():
    task_cfg = registry.get_eval_task('alei_asr')

    t = EvalTask(dataset=registry.get_dataset('KeSpeech'),
                 prompt=registry.get_prompt('KeSpeech'),
                 predictor=registry.get_model(task_cfg.model),
                 evaluator=registry.get_evaluator(task_cfg.evaluator),
                 post_process=[registry.get_process(item) for item in task_cfg.post_process],
                 agg=registry.get_agg(task_cfg.agg),
                 recorder=Recorder('log/KeSpeech.jsonl'))
    res = t.run()
    print(res)
