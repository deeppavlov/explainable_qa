import re


def kbqa_accuracy(pred_answer_labels_batch, pred_answer_ids_batch, pred_query_batch,
                  gold_answer_labels_batch, gold_answer_ids_batch, gold_query_batch) -> float:
    num_samples = len(pred_answer_ids_batch)
    correct = 0
    for pred_answer_label, pred_answer_ids, pred_query, gold_answer_labels, gold_answer_ids, gold_query in \
            zip(pred_answer_labels_batch, pred_answer_ids_batch, pred_query_batch,
                gold_answer_labels_batch, gold_answer_ids_batch, gold_query_batch):
        found_date = False
        if pred_answer_ids and gold_answer_ids and re.findall(r"[\d]{3,4}", pred_answer_ids[0]) and \
                re.findall(r"[\d]{3,4}", pred_answer_ids[0]) == re.findall(r"[\d]{3,4}", gold_answer_ids[0]):
            found_date = True
        found_label = False
        if len(gold_answer_labels) == 1 and len(pred_answer_label) > 1 and pred_answer_label == gold_answer_labels[0]:
            found_label = True
        if set(pred_answer_ids) == set(gold_answer_ids) or pred_query == gold_query or found_date or found_label:
            correct += 1

    acc = correct / num_samples if num_samples else 0
    return acc


def kbqa_f1(question_batch, pred_answer_ids_batch, pred_query_batch, gold_answer_ids_batch, gold_query_batch) -> float:
    precision, recall, f1 = 0, 0, 0
    for question, pred_answer_ids, pred_query, gold_answer_ids, gold_query in \
            zip(question_batch, pred_answer_ids_batch, pred_query_batch, gold_answer_ids_batch, gold_query_batch):
        gold_answer_ids = [gold_ans_id.split("/")[-1].replace("@en", "").strip('"') for gold_ans_id in gold_answer_ids
                           if isinstance(gold_ans_id, str)]
        inters = set(pred_answer_ids).intersection(set(gold_answer_ids))
        num_pred = len(set(pred_answer_ids))
        num_gold = len(set(gold_answer_ids))
        if pred_query == gold_query or set(pred_answer_ids) == set(gold_answer_ids):
            precision += 1.0
            recall += 1.0
        else:
            if num_pred > 0:
                precision += 1.0 * len(inters) / num_pred
            if num_gold > 0:
                recall += 1.0 * len(inters) / num_gold
    precision /= len(question_batch)
    recall /= len(question_batch)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
