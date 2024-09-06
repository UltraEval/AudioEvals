
# News/Updates:

2024/9/6 🎉 we support `Qwen/Qwen2-Audio-7B`, `Qwen/Qwen2-Audio-7B-Instruct` models! 

# OVERVIEW

AudioEvals is an open-source framework designed for the evaluation of large audio models (Audio LLMs).
With this tool, you can easily evaluate any Audio LLM in one go.

Not only do we offer a ready-to-use solution that includes a collection of
audio benchmarks and evaluation methodologies, but we also provide the capability for
you to customize your evaluations.


# Quick Start

## ready env
```shell
git clone https://github.com//AduioEval.git
cd AduioEval
conda create -n aduioeval python=3.10 -y
conda activate aduioeval
pip install -r requirements.txt
```

## run
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir log/
# eval gemini model only when you are in USA
export GOOGLE_API_KEY=$your-key
python audio_evals/main.py --dataset KeSpeech-sample --model gemini-pro

# eval qwen-audio api model
export DASHSCOPE_API_KEY=$your-key
python audio_evals/main.py --dataset KeSpeech-sample --model qwen-audio

# eval qwen2-audio  offline model in local
pip install -r requirments-offline-model.txt
python audio_evals/main.py --dataset KeSpeech-sample --model qwen2-audio-offline
```

## res

after program executed, you will get the performance in console and detail result as below:

```txt
- res
    |-- $time-$name-$dataset.jsonl
```

# performance

![assets/performance.png](assets/performance.png)


> () is offical performance 

| model      | multilingual_librispeech          | librispeech                                                                             | FLEURS | covost2                                                                                            | KeSpeech         | WenetSpeech                       | ClothoAQA | AISHELL-1 |
|------------|-----------------------------------|-----------------------------------------------------------------------------------------|-----|----------------------------------------------------------------------------------------------------|------------------|-----------------------------------|-----------|-----------|
| qwen-audio | mls_french 32.84 mls_german 49.07 | dev-clean 1.85(1.8);  dev-other 4.14(4.0)  test-clean   2.21(2.0), test-other 4.27(4.2) | 25 | en-zh 41.24（41.5）, zh-en 15.93（15.7）, en-de 25.13（25.1）;en-ar 0.16; en-ca 1.5, en-cy 1.2 , en-et 0.67 | 6.6 | test_meeting 11.23, test_net 9.25 | 58.86（57.9） | （1.3） |




# Usage

![assets/img_1.png](assets/img_1.png)

To run the evaluation script, use the following command:

```bash
python audio_evals/main.py --dataset <dataset_name> --model <model_name>
```

## Dataset Options

The `--dataset` parameter allows you to specify which dataset to use for evaluation. The following options are available:

- `KeSpeech`
- `librispeech-test-clean`
- `librispeech-dev-clean`
- `librispeech-test-other`
- `librispeech-dev-other`
- `mls_dutch`
- `mls_french`
- `mls_german`
- `mls_italian`
- `mls_polish`
- `mls_portuguese`
- `mls_spanish`
- `fleurs-zh`
- `covost2-en-ar`
- `covost2-en-ca`
- `covost2-en-cy`
- `covost2-en-de`
- `covost2-en-et`
- `covost2-en-fa`
- `covost2-en-id`
- `covost2-en-ja`
- `covost2-en-lv`
- `covost2-en-mn`
- `covost2-en-sl`
- `covost2-en-sv`
- `covost2-en-ta`
- `covost2-en-tr`
- `covost2-en-zh`
- `covost2-zh-en`
- `WenetSpeech-test-meeting`
- `WenetSpeech-test-net`

### support dataset detail
| <dataset_name> | name                     | domain                            | metric |
|--------------|--------------------------|-----------------------------------|--------|
| clotho-aqa   | ClothoAQA                | QAQ(AudioQA)                      | acc    |
| mls-*        | multilingual_librispeech | ASR(Automatic Speech Recognition) | wer    |
| KeSpeech     | KeSpeech | ASR | cer    |
| librispeech-* | librispeech              | ASR                               | wer    |
| fleurs-*     | FLEURS                   | ASR                               | wer    |
| aisheel1     | AISHELL-1                | ASR                               | wer    |
| WenetSpeech-* | WenetSpeech              | ASR                               | wer    |
| covost2-*    | covost2                  | STT(Speech Text Translation)      | BLEU   |

eval your dataset: [docs/how add a dataset.md](docs%2Fhow%20add%20a%20dataset.md)


### Model Options

The `--model` parameter allows you to specify which model to use for evaluation. The following options are available:

- `qwen2-audio`: Use the Qwen2 Audio model.
- `gemini-pro`: Use the Gemini 1.5 Pro model.
- `gemini-1.5-flash`: Use the Gemini 1.5 Flash model.
- `qwen-audio`: Use the qwen2-audio-instruct Audio API model.

eval your model: [docs/how eval your model.md](docs%2Fhow%20eval%20your%20model.md)

# Contact us
If you have questions, suggestions, or feature requests regarding AudioEvals, please submit GitHub Issues to jointly build an open and transparent UltraEval evaluation community.


# Citation
