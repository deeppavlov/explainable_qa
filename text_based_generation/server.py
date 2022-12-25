import copy
import json
import logging
import os
import re
import requests

import nltk
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from deeppavlov import build_model
from metrics import squad_exact_match, squad_f1
from sacrebleu.metrics import BLEU


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

nltk.download('stopwords')

PORT = int(os.getenv("PORT"))
RETRIEVE_ENDPOINT = os.getenv("RETRIEVE_ENDPOINT")

re_tokenizer = re.compile(r"[\w']+|[^\w ]")
SacreBLEU = BLEU()
stopwords = set(stopwords.words("russian"))

config_name = "mt5_long_ans.json"

try:
    generator = build_model(config_name, download=True)
    logger.info("generation model is loaded.")
except Exception as e:
    logger.exception(e)
    raise e


def add_punctuation(sentence):
    sentence = sentence.strip()
    if sentence[-1] not in ".!?":
        sentence = f"{sentence}."
    return sentence


@app.route("/generate", methods=["POST"])
def generate():
    questions_batch = request.json.get("questions", [])
    paragraphs_batch = request.json.get("paragraphs", [])
    res, _ = generator(questions_batch, paragraphs_batch)
    long_explanations = []
    for long_exp in res:
        long_exp = add_punctuation(long_exp)
        long_explanations.append({"long_explanation": long_exp})
    return jsonify(long_explanations)


@app.route("/ans_expl", methods=["POST"])
def get_answer_and_explaination():
    questions_batch = request.json.get("questions", [])
    retrieve_res_batch = requests.post(RETRIEVE_ENDPOINT, json={"questions": questions_batch}).json()
    results_batch = [[] for _ in questions_batch]
    if retrieve_res_batch:
        results_batch = []
        for question, retrieve_res_list in zip(questions_batch, retrieve_res_batch):
            questions = [question for _ in retrieve_res_list]
            sentences = [elem["answer_sentence"] for elem in retrieve_res_list]
            long_expl_list, _ = generator(questions, sentences)
            answers = [elem["answer"] for elem in retrieve_res_list]
            answer_scores = [elem["answer_score"] for elem in retrieve_res_list]
            answer_places = [elem["answer_place"] for elem in retrieve_res_list]
            long_expl_list = [add_punctuation(expl) for expl in long_expl_list]
            results_list = []
            for long_expl, answer, answer_score, answer_place in \
                    zip(long_expl_list, answers, answer_scores, answer_places):
                answer_tokens = re.findall(re_tokenizer, answer)
                answer_tokens = [tok for tok in answer_tokens if len(tok) > 0 and tok not in stopwords]
                if any([tok in long_expl for tok in answer_tokens]):
                    results_list.append({"answer": answer, "answer_score": answer_score, "answer_place": answer_place,
                                         "long_explanation": long_expl})
            if not results_list:
                for long_expl, answer, answer_score, answer_place, sentence in \
                    zip(long_expl_list, answers, answer_scores, answer_places, sentences):
                    results_list.append({"answer": answer, "answer_score": answer_score,
                                         "answer_place": answer_place, "long_explanation": sentence})
            results_batch.append(results_list)
    return jsonify(results_batch)


@app.route("/get_metrics_expl", methods=["POST"])
def get_metrics_expl():
    num_samples = request.json.get("num_samples", 100)
    with open("/root/.deeppavlov/downloads/dsberquad/sbersquad_detailed.json", 'r') as fl:
        dataset = json.load(fl)

    if num_samples == "all":
        num_samples = len(dataset["test"])
    long_answers = []
    res_answers = []
    batch_size = 20
    num_batches = len(dataset["test"][:num_samples]) // batch_size + int(len(dataset["test"][:num_samples]) % batch_size > 0)
    logger.info(f"num_samples {num_samples} -- num_batches {num_batches}")
    for i in range(num_batches):
        q_batch = []
        c_batch = []
        l_answer = []
        for (question, contexts), (short_answer, long_answer) in dataset["test"][i*batch_size:(i+1)*batch_size]:
            q_batch.append(question)
            c_batch.append(contexts)
            long_answers.append(long_answer)
        res, _ = generator(q_batch, c_batch)
        for output_answer in res:
            output_answer = output_answer.replace("<pad>", "").replace("</s>", "").strip()
            res_answers.append(copy.deepcopy(output_answer))
        if i % 10 == 0:
            logger.info(f"number of testing batch: {i}")
    
    generation_sacrebleu = SacreBLEU.corpus_score(res_answers, [long_answers]).score
    return jsonify({"SacreBLUE of explanation generation": generation_sacrebleu})


@app.route("/get_metrics_ans_expl", methods=["POST"])
def get_metrics_ans_expl():
    num_samples = request.json.get("num_samples", 100)
    with open("/root/.deeppavlov/downloads/dsberquad/sbersquad_detailed.json", 'r') as fl:
        dataset = json.load(fl)

    short_answers, res_short_answers = [], []
    long_answers, res_long_answers = [], []
    for (question, contexts), (short_answer, long_answer) in dataset["test"][:num_samples]:
        retrieve_res_batch = requests.post(RETRIEVE_ENDPOINT, json={"questions": [question]}).json()
        if retrieve_res_batch and retrieve_res_batch[0]:
            retrieve_res_list = retrieve_res_batch[0]
            res_short_answer = retrieve_res_list[0]["answer"]
            short_answers.append([short_answer])
            res_short_answers.append(res_short_answer)
            sentences = [retrieve_res_list[0]["answer_sentence"]]
            res, _ = generator([question], [sentences])
            if res:
                output_answer = res[0].replace("<pad>", "").replace("</s>", "").strip()
                long_answers.append([long_answer])
                res_long_answers.append(copy.deepcopy(output_answer))

    exact_match = squad_exact_match(short_answers, res_short_answers)
    f1 = squad_f1(short_answers, res_short_answers)
    ans_expl_sacrebleu = SacreBLEU.corpus_score(res_long_answers, long_answers).score
    return jsonify({"SacreBLEU of answer search and subsequent explanation generation": ans_expl_sacrebleu,
                    "F1 of short answers": f1,
                    "Exact match of short answers": exact_match})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
