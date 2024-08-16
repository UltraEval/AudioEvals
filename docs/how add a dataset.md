# how add a dataset in AudioEvals?


In practice, you may need eval your custom audio dataset.


here are steps:


## JSON file: 
1. make sure your dataset file is `jsonl` format
2. new a file `**.yaml`
    content like :
    ```yaml
   aleil-local:  # name after cli: --dataset $name
   class: audio_evals.dataset.dataset.RelativeASR
   args:  
     default_task: alei_asr
     f_name: /Users/a1/project/UltraEval/dataset/mb-temp-model-reuse-aqa/data/mb-temp-model-reuse-aqa.jsonl
     ref_col: 标准答案
    ```