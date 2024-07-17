import argparse
import logging
from datetime import datetime

from audio_evals.eval_task import EvalTask
from audio_evals.recorder import Recorder
from audio_evals.registry import registry
from audio_evals.utils import put_to_hdfs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='KeSpeech')
    parser.add_argument('--model', default='qwen-audio')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--evaluator', default='')
    parser.add_argument('--agg', default='')
    parser.add_argument('--post_process', default='')
    parser.add_argument('--save', default='')

    args = parser.parse_args()
    args.post_process = args.post_process.split()
    return args


def main():
    time_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f'app-{time_id}.log'), logging.StreamHandler()])
    args = get_args()
    if not args.save:
        args.save = f'log/{time_id}.jsonl'

    dataset = registry.get_dataset(args.dataset)
    task_cfg = registry.get_eval_task(dataset.task_name)

    attrs = dir(task_cfg)
    for attr in dir(args):
        if not attr.startswith('__') and attr in attrs and getattr(args, attr):
            setattr(task_cfg, attr, getattr(args, attr))

    t = EvalTask(dataset=dataset,
                 prompt=registry.get_prompt(task_cfg.prompt),
                 predictor=registry.get_model(task_cfg.model),
                 evaluator=registry.get_evaluator(task_cfg.evaluator),
                 post_process=[registry.get_process(item) for item in task_cfg.post_process],
                 agg=registry.get_agg(task_cfg.agg),
                 recorder=Recorder(args.save))
    res = t.run()
    print(res[0])
    put_to_hdfs(args.save, '/user/tc_agi/AudioEvals/log/')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
