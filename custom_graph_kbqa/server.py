import logging
import os
import json

from flask import Flask, jsonify, request
from deeppavlov import build_model


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

PORT = int(os.getenv("PORT"))
config_name = "kbqa_custom_graph.json"

try:
    retriever = build_model(config_name, download=True)
    logger.info("retriever model is loaded.")
except Exception as e:
    logger.exception(e)
    raise e


def get_result(request):
    questions_batch = request.json.get("questions", [])
    answers = retriever(questions_batch)
    return answers


@app.route("/respond", methods=["POST"])
def respond():
    result = get_result(request)
    return jsonify(result)


@app.route("/get_metrics", methods=["POST"])
def get_metrics():
    def check(output, y):
        for ans in output:
            if ans.replace('http://rdf.freebase.com/ns/', '') in y:
                return True
        return False
    
    with open('~/.deeppavlov/downloads/custom_graph/grail_qa.json', 'r') as f:
        grail_qa = json.load(f)

    correct_gqa = 0

    for example in grail_qa:
        question = example['question']
        answer_argument = example['answer']['answer_argument']

        output = retriever([question])[0][0]
        if check(output, answer_argument):
            correct_gqa += 1

    return correct_gqa / len(grail_qa)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
