import json
import logging
import os
import re
import requests

import nltk
from flask import Flask, jsonify, request
from deeppavlov import build_model
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

PORT = int(os.getenv("PORT"))
KBQA_ENDPOINT = os.getenv("KBQA_ENDPOINT")
CORPUS_BLEU = 0

re_tokenizer = re.compile(r"[\w']+|[^\w ]")

config_name = "graph2text_bart_infer.json"

try:
    generator = build_model(config_name, download=True)
    logger.info("generation model is loaded.")
except Exception as e:
    logger.exception(e)
    raise e


alphabet_rus = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
alphabet_eng = "abcdefghijklmnopqrstuvwxyz"

@app.route("/generate", methods=["POST"])
def generate():
    triplets_batch = request.json.get("triplets", [])
    texts = generator(triplets_batch)
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
            for answer_info in kbqa_res_batch:
                triplets = answer_info["triplets"]
                long_expl = generator([triplets])
                answer_info["long_explanation"] = long_expl[0]
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
