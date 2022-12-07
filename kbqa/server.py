import json
import logging
import os

from flask import Flask, jsonify, request
from deeppavlov import build_model
from models.kbqa.query_generator import QueryFormatter
from utils.metrics import kbqa_accuracy, kbqa_f1
from utils.sq_reader import RuBQReader, LCQuADReader
from utils.preprocessors import get_ent_rels


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

LAN = str(os.getenv("LAN"))
PORT = int(os.getenv("PORT"))

if LAN == "RU":
    config_name = "kbqa_rubq.json"
elif LAN == "EN":
    config_name = "kbqa_lcquad2.json"

try:
    kbqa = build_model(config_name, download=False)
    logger.info(f"kbqa model is loaded: {kbqa}")
except Exception as e:
    logger.exception(e)
    raise e


alphabet_rus = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
alphabet_eng = "abcdefghijklmnopqrstuvwxyz"

@app.route("/respond", methods=["POST"])
def respond():
    questions_batch = request.json.get("questions", [])
    num_rus_letters, num_eng_letters = 0, 0
    if questions_batch:
        for letter in questions_batch[0].lower():
            if letter in alphabet_rus:
                num_rus_letters += 1
            elif letter in alphabet_eng:
                num_eng_letters += 1
    if (LAN == "RU" and num_rus_letters > num_eng_letters) or (LAN == "EN" and num_eng_letters > num_rus_letters):
        answer_info_batch = []
        try:
            answers_batch, confs_batch, answer_ids_batch, ent_rels_batch, query_batch, triplets_batch = \
                kbqa(questions_batch)
            for answer, confs, answer_ids, ent_rels, query, triplets in \
                    zip(answers_batch, confs_batch, answer_ids_batch, ent_rels_batch, query_batch, triplets_batch):
                answer_info_batch.append({"answer": answer, "confidence": confs, "answer_ids": answer_ids,
                                          "entities_and_rels": ent_rels, "sparql_query": query, "triplets": triplets})
        except Exception as e:
            logger.info(f"Error in /respond {e}")
    else:
        if LAN == "RU":
            answer_info_batch = ["!!! Warning: you are passing an English question, but the Russian model is loaded."
                                 "Please specify in docker-compose.yml for kbqa service LAN: EN"]
        elif LAN == "EN":
            answer_info_batch = ["!!! Warning: you are passing an Russian question, but the English model is loaded."
                                 "Please specify in docker-compose.yml for kbqa service LAN: RU"]
    logger.info(f"answer info {answer_info_batch[0]}")
    return jsonify(answer_info_batch)


@app.route("/get_metrics", methods=["POST"])
def get_metrics():
    num_samples = request.json.get("num_samples", 100)
    if LAN == "RU":
        dataset_reader = RuBQReader()
        query_formatter = QueryFormatter(query_info = {"unk_var": "?answer", "mid_var": "?ent"})
        dataset = dataset_reader.read("/root/.deeppavlov/downloads/rubq/rubq2.0.json", num_samples=num_samples)
        logger.info(f"samples for testing: {len(dataset['test'])}")
        res_answers, res_answer_ids, res_queries = [], [], []
        gold_answers, gold_answer_ids, gold_queries, question_types = [], [], [], []
        correct_expl = 0
        for n, ((question, question_type), (g_answer_ids, g_answer_labels, g_query)) \
                in enumerate(dataset["test"][:num_samples]):
            g_entities, g_rels = get_ent_rels(g_query, "wikidata")
            g_query = query_formatter([g_query])[0]
            logger.info(f"{n} --- question {question}")
            answers_batch, confs_batch, answer_ids_batch, ent_rels_batch, query_batch, triplets_batch = kbqa([question])
            gold_answers.append(g_answer_labels)
            gold_answer_ids.append(g_answer_ids)
            gold_queries.append(g_query)
            question_types.append(question_type)
            res_answers.append(answers_batch[0])
            res_answer_ids.append(answer_ids_batch[0])
            res_queries.append(query_batch[0])
            expl_corr_flag = False
            if answer_ids_batch and ent_rels_batch and len(ent_rels_batch[0]) == 2:
                f_entities, f_rels = ent_rels_batch[0]
                if set(answer_ids_batch[0]) == set(g_answer_ids) and set(f_entities) == set(g_entities) \
                        and set(f_rels) == set(g_rels):
                    expl_corr_flag = True
            if expl_corr_flag:
                correct_expl += 1

        qa_acc = kbqa_accuracy(res_answers, res_answer_ids, res_queries, gold_answers, gold_answer_ids,
                               gold_queries)

        acc_by_types = {}
        for type_name, include_types, not_include_type in [["with answer", [], "no-answer"],
                                                           ["1-hop", ["1-hop"], ""],
                                                           ["1-hop + reverse", ["1-hop", "reverse"], ""],
                                                           ["1-hop + count", ["1-hop", "count"], ""],
                                                           ["1-hop + exclusion", ["1-hop", "exclusion"], ""],
                                                           ["multi-constraint", ["multi-constraint"], ""],
                                                           ["qualifier-constraint", ["qualifier-constraint"], ""],
                                                           ["no-answer", ["no-answer"], ""]
                                                           ]:
            answers_info = list(zip(question_types, gold_answers, gold_answer_ids, gold_queries, res_answers,
                                    res_answer_ids, res_queries))
            answers_info = [elem for elem in answers_info
                            if ((not include_types or all([tp in elem[0] for tp in include_types])) and
                                (not not_include_type or not_include_type not in elem[0]))]

            cur_gold_answers = [elem[1] for elem in answers_info]
            cur_gold_answer_ids = [elem[2] for elem in answers_info]
            cur_gold_queries = [elem[3] for elem in answers_info]
            cur_res_answers = [elem[4] for elem in answers_info]
            cur_res_answer_ids = [elem[5] for elem in answers_info]
            cur_res_queries = [elem[6] for elem in answers_info]
            cur_acc = kbqa_accuracy(cur_res_answers, cur_res_answer_ids, cur_res_queries, cur_gold_answers,
                                    cur_gold_answer_ids, cur_gold_queries)
            acc_by_types[type_name] = round(cur_acc, 4)
        
        expl_acc = correct_expl / len(dataset["test"])
        return jsonify({"question_answering_accuracy": round(qa_acc, 4),
                        "accuracy_of_explanations": round(expl_acc, 4),
                        "question_answering_accuracy_by_types": acc_by_types})

    elif LAN == "EN":
        kbqa_lcquad = build_model("kbqa_lcquad.json")
        dataset_reader = LCQuADReader()
        query_formatter = QueryFormatter(query_info = {"unk_var": "?uri", "mid_var": "?x"})
        dataset = dataset_reader.read("/root/.deeppavlov/downloads/lcquad/lcquad.json", num_samples=num_samples)
        logger.info(f"samples for testing: {len(dataset['test'])}")
        questions = []
        res_answer_ids, res_queries = [], []
        gold_answer_ids, gold_queries = [], []
        correct_expl = 0
        for n, (question, (g_answer_ids, g_query)) in enumerate(dataset["test"][:num_samples]):
            g_entities, g_rels = get_ent_rels(g_query, "dbpedia")
            g_query = query_formatter([g_query])[0]
            logger.info(f"{n} --- question {question}")
            answers_batch, confs_batch, answer_ids_batch, ent_rels_batch, query_batch, triplets_batch = \
                kbqa_lcquad([question])
            questions.append(question)
            gold_answer_ids.append(g_answer_ids)
            gold_queries.append(g_query)
            res_answer_ids.append(answer_ids_batch[0])
            res_queries.append(query_batch[0])
            if answer_ids_batch and ent_rels_batch and len(ent_rels_batch[0]) == 2:
                f_entities, f_rels = ent_rels_batch[0]
                g_answer_ids = [gold_ans_id.split("/")[-1].replace("@en", "").strip('"')
                                for gold_ans_id in g_answer_ids if isinstance(gold_ans_id, str)]
                g_entities = [ent.split("/")[-1] for ent in g_entities]
                f_rels = [rel.split("/")[-1] for rel in f_rels]
                g_rels = [rel.split("/")[-1] for rel in g_rels]

                if set(answer_ids_batch[0]) == set(g_answer_ids) and set(f_entities) == set(g_entities) \
                        and set(f_rels) == set(g_rels):
                    correct_expl += 1

        precision, recall, f1 = kbqa_f1(questions, res_answer_ids, res_queries, gold_answer_ids, gold_queries)
        expl_acc = correct_expl / len(dataset["test"][:num_samples])
        return jsonify({"qa_precision": round(precision, 4), "qa_recall": round(recall, 4), "qa_f1": round(f1, 4),
                        "accuracy_of_explanations": round(expl_acc, 4)})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
