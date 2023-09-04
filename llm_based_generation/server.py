import os
import re
import json
import logging
from tqdm import trange
import requests

import torch
import nltk
from flask import Flask, jsonify, request
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
    )

from utils import generate_output


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

PORT = int(os.getenv("PORT"))
KBQA_ENDPOINT = os.getenv("KBQA_ENDPOINT")
CORPUS_BLEU = 0
model_name = os.getenv("MODEL_NAME", "bigscience/bloomz-7b1")
quantized = bool(os.getenv("QUANTIZED", 1))
prompt_style = os.getenv("PROMPT_STYLE", "fewshot_triplets")


with open("generation_params.json", "r") as f:
    generation_params = json.load(f)
    generation_config = GenerationConfig(**generation_params)

re_tokenizer = re.compile(r"[\w']+|[^\w ]")
alphabet_rus = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
alphabet_eng = "abcdefghijklmnopqrstuvwxyz"


def load_models_and_params(hf_hub_name: str):
    default_params = {"sep_toks": None, "roles": ["USER", "ASSISTANT"], "stop_str": "USER", "type": "other"}
    mapping_models = {
    "lmsys/vicuna-13b-v1.3": {"sep_toks": [" ", "</s>"], "roles": ["USER", "ASSISTANT"], "stop_str": "</s>", "type": "vicuna"},
    "OpenAssistant/pythia-12b-sft-v8-7k-steps": {"sep_toks": ["<|endoftext|>"], "roles": ["<|prompter|>", "<|assistant|>"], "stop_str": None, "type": "pythia"},
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_hub_name)
    if quantized:
        model = AutoModelForCausalLM.from_pretrained(hf_hub_name, quantization_config=bnb_config, trust_remote_code=True, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_hub_name, device_map="auto")

    if torch.__version__ >= "2":
        model = torch.compile(model)
    params = mapping_models.get(hf_hub_name, default_params)
    return model, tokenizer, params


try:
    model, tokenizer, params = load_models_and_params(model_name)
    logger.info("generation model is loaded.")
except Exception as e:
    logger.exception(e)
    raise e


@app.route("/generate", methods=["POST"])
def generate():
    triplets_batch = request.json.get("triplets", [])
    questions_batch = request.json.get("questions", ["" for _ in range(len(triplets_batch))])
    texts = []
    for triplets, question in trange(zip(triplets_batch, questions_batch)):
        text = generate_output(model, tokenizer, params, triplets, question, prompt_style, generation_config)
        texts.append(text)
    return jsonify(texts)


@app.route("/ans_expl", methods=["POST"])
def get_answer_and_explanation():
    questions_batch = request.json.get("questions", [])
    num_rus_letters, num_eng_letters = 0, 0
    if questions_batch:
        for letter in questions_batch[0].lower():
            if letter in alphabet_rus:
                num_rus_letters += 1
            elif letter in alphabet_eng:
                num_eng_letters += 1
    if num_rus_letters > num_eng_letters:
        answer_info_batch = ["!!! Warning: the generative model supports only English. "
                             "Please enter the question in English and check that you have specified in "
                             "docker-compose.yml for kbqa service LAN: EN"]
    else:
        answer_info_batch = []
        try:
            kbqa_res_batch = requests.post(KBQA_ENDPOINT, json={"questions": questions_batch}).json()
            for (answer_info, question) in zip(kbqa_res_batch, questions_batch):
                triplets = answer_info["triplets"]
                long_expl = generate_output(model, tokenizer, params, triplets, question, prompt_style, generation_config)
                answer_info["long_explanation"] = long_expl
                answer_info_batch.append(answer_info)
        except Exception as e:
            logger.info(f"Error in /ans_expl {e}")
    logger.info(f"answer info {answer_info_batch[0]}")
    return jsonify(answer_info_batch)


@app.route("/get_metrics", methods=["POST"])
def get_metrics():
    logger.info("---------------- getting metrics")
    with open("/root/.deeppavlov/downloads/lcquad/lcquad_long_ans.json", 'r') as fl:
        dataset = json.load(fl)
    res_answers_list, gold_answers_list = [], []
    scores1_list, scores2_list = [], []
    for n, ((question, triplets), init_gold_answers) in enumerate(dataset):
        try:
            texts = generator([triplets])
            res_answers_list.append(nltk.word_tokenize(texts[0]))
            gold_answers = [nltk.word_tokenize(gold_ans) for gold_ans in init_gold_answers]
            gold_answers_list.append(gold_answers)

            long_answer = re.findall(re_tokenizer, texts[0].lower())
            gold_answers = [re.findall(re_tokenizer, gold_answer.lower()) for gold_answer in init_gold_answers]
            scores1_list.append(sentence_bleu(gold_answers, long_answer, weights=(1, 0, 0, 0)))
            scores2_list.append(sentence_bleu(gold_answers, long_answer, weights=(0.5, 0.5, 0, 0)))
        except Exception as e:
            logger.info(f"Error in /get_metrics {e}")
        if n % 5 == 0:
            logger.info(f"num testing sample: {n}")

    if CORPUS_BLEU:
        bleu1 = corpus_bleu(gold_answers_list, res_answers_list, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(gold_answers_list, res_answers_list, weights=(0.5, 0.5, 0, 0))
    else:
        bleu1 = sum(scores1_list) / len(scores1_list)
        bleu2 = sum(scores2_list) / len(scores2_list)
    logger.info(f"BLEU scores: {bleu1} --- {bleu2}")
    return jsonify({"BLEU-1": bleu1, "BLEU-2": bleu2})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
