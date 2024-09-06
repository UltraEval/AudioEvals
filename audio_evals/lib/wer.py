import editdistance as ed

from audio_evals.lib.evaluate_tokenizer import EvaluationTokenizer
from audio_evals.lib.text_normalization.basic import BasicTextNormalizer
from audio_evals.lib.text_normalization.cn_tn import TextNorm
from audio_evals.lib.text_normalization.en import EnglishTextNormalizer


english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
                to_banjiao=False,
                to_upper=False,
                to_lower=False,
                remove_fillers=False,
                remove_erhua=False,
                check_chars=False,
                remove_space=False,
                cc_mode="",
            )
basic_normalizer = BasicTextNormalizer()


def compute_wer(refs, hyps, language="en"):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=False,
        character_tokenization=False,
    )
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]

        ref = english_normalizer(ref)
        pred = english_normalizer(pred)
        if language in ["zh"]:
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)

        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        if language == "zh":
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
        if i == 0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
            print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance / ref_length
