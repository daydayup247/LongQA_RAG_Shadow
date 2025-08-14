import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score
)

dataset2metric = {
    "qasper": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score
}


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        # prediction = prediction.lstrip('\n')
        # print(prediction)
        # prediction = prediction.strip()
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == '__main__':
    scores = dict()
    path = f"pred/llama2-7b-chat-4k-discriminator/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score

    out_path = f"pred/llama2-7b-chat-4k-discriminator/result.json"

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
