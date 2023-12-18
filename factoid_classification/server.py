import os
import json
import pickle
import logging
import numpy as np
from sklearn.metrics import f1_score

from flask import Flask, jsonify, request
from deeppavlov import build_model


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = Flask(__name__)

PORT = int(os.getenv("PORT"))

try:
    model = build_model("factoid_clf.json", download=True)
    logger.info(f"the model is loaded successfully!")
except Exception as e:
    logger.exception(e)
    raise e



@app.route("/predict", methods=["POST"])
def respond():
    questions_batch = request.json.get("questions", [])
    logger.info(f"questions_batch {questions_batch}")
    preds = model(questions_batch)
    res = {"class": [], "probability": []}
    for pred in preds:
        if pred[0] > 0.5:
            res['class'].append("factoid")
            res['probability'].append(pred[0].item())
        else:
            res['class'].append("non-factoid")
            res['probability'].append(pred[1].item())
    return jsonify(res)

@app.route("/get_metrics", methods=["POST"])
def get_metrics_factoid():
    with open("/root/.deeppavlov/downloads/factoid_clf/factoid_classification.pickle", 'rb') as f:
        dataset = pickle.load(f)

    logger.info(f"samples for testing: {len(dataset['test'])}")
    batch_size = 32
    preds_batch_factoid = []
    golds_batch_factoid = []
    for batch in range(0, len(dataset["test"]), batch_size):
        data = dataset["test"][batch : batch+batch_size]
        input_text, gold_labels = list(zip(*data))
        preds = model(input_text)
        preds_batch_factoid.extend([np.argmax(p) for p in preds])
        golds_batch_factoid.extend([g == "non_factoid" for g in gold_labels])  
    f1_factoid = f1_score(golds_batch_factoid, preds_batch_factoid)
    return jsonify({"f1_factoid": f1_factoid})
