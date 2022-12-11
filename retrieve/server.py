import logging
import os

from flask import Flask, jsonify, request
from deeppavlov import build_model


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

PORT = int(os.getenv("PORT"))
LITE_INDEX = int(os.getenv("LITE_INDEX"))

if LITE_INDEX:
    config_name = "retriever_lite.json"
else:
    config_name = "retriever.json"

try:
    retriever = build_model(config_name, download=True)
    logger.info("retriever model is loaded.")
except Exception as e:
    logger.exception(e)
    raise e


def get_result(request):
    questions_batch = request.json.get("questions", [])
    answers_batch, answer_scores_batch, answer_logits_batch, answer_places_batch, answer_sentences_batch = \
        retriever(questions_batch)
    answer_info_batch = []
    for answers, answer_scores, answer_logits, answer_places, answer_sentences in \
            zip(answers_batch, answer_scores_batch, answer_logits_batch, answer_places_batch, answer_sentences_batch):
        answer_info_list = []
        for answer, answer_score, answer_logit, answer_place, answer_sentence in \
                zip(answers, answer_scores, answer_logits, answer_places, answer_sentences):
            answer_info_list.append({"answer": answer, "answer_score": answer_score, "answer_logit": answer_logit,
                                     "answer_place": answer_place, "answer_sentence": answer_sentence})
        answer_info_batch.append(answer_info_list)
    logger.info(f"answer info {answer_info_batch[0][:2]}")
    return answer_info_batch


@app.route("/respond", methods=["POST"])
def respond():
    result = get_result(request)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
