#!/usr/bin/env python3
"""
인기도 기준선(baseline) + 재사용 가능한 평가 하니스(harness).

이 파일의 핵심은 인기도 점수 자체가 아니라 하니스다.
`evaluate(score_fn, ...)`는 사용자 history를 아이템의 랭킹 리스트로 매핑하는
임의의 함수를 받아 Recall@k / NDCG@k를 반환한다. 앞으로의 모든 모델
(시간 가중 풀링, 텍스트 임베딩 검색, SASRec, 크로스 도메인, ...)은 자신만의
`score_fn`을 제공함으로써 동일한(SAME) 하니스에 끼워 넣는다. 이렇게 해야
프로젝트 전체에서 비교를 동등한(apples-to-apples) 조건으로 유지할 수 있다.

평가 프로토콜:
- Leave-one-out: 각 사용자는 정확히 하나의 held-out 타깃 아이템을 가진다.
- 모든 아이템에 대한 전체(FULL) 랭킹 (샘플링된 negative가 아님). 샘플링된
  메트릭은 모델 비교를 왜곡하는 것으로 알려져 있으므로(Krichene & Rendle, 2020),
  전체 아이템 집합을 대상으로 랭킹한다. 인기도의 경우 이 비용은 저렴하다.
- 사용자의 입력 history에 이미 있는 아이템은 추천에서 제외한다.
- 여기서 Recall@k == Hit@k (단일 타깃): 타깃이 top-k에 나타나면 1.
- NDCG@k: 1/log2(rank+1), 여기서 rank는 top-k 내 타깃의 1-기반(1-indexed) 위치.
"""

import argparse
import json
import math
import os
from collections import Counter


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def make_popularity_score_fn(item_pop):
    """인기도 내림차순으로 정렬된 score_fn(history) -> list[(item, score)]를 반환한다.

    인기도는 사용자의 history를 전혀 사용하지 않지만(그것이 이 기준선의 핵심이다),
    다른 모든 모델의 score_fn과 시그니처를 맞추기 위해 history는 그대로 전달받는다.
    """
    # precompute the global popularity ranking once
    ranked = sorted(item_pop.items(), key=lambda kv: kv[1], reverse=True)

    def score_fn(history):
        return ranked  # already sorted; harness will filter seen items
    return score_fn


def evaluate(score_fn, eval_examples, k=10, exclude_seen=True):
    """eval_examples에 대해 Recall@k와 NDCG@k를 계산한다.

    score_fn(history)은 내림차순으로 정렬된 (item, score)의 iterable을 반환해야 한다.
    """
    recall_sum = 0.0
    ndcg_sum = 0.0
    n = 0
    for ex in eval_examples:
        target = ex["target"]
        if target is None:
            continue
        seen = set(ex["history"]) if exclude_seen else set()

        # take the top-k items not already seen by the user
        topk = []
        for item, _ in score_fn(ex["history"]):
            if item in seen:
                continue
            topk.append(item)
            if len(topk) >= k:
                break

        n += 1
        if target in topk:
            rank = topk.index(target) + 1  # 1-indexed
            recall_sum += 1.0
            ndcg_sum += 1.0 / math.log2(rank + 1)

    return {
        "n_eval": n,
        f"Recall@{k}": round(recall_sum / n, 4) if n else 0.0,
        f"NDCG@{k}": round(ndcg_sum / n, 4) if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir produced by preprocess.py")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "item_pop.json")) as fp:
        item_pop = Counter(json.load(fp))
    eval_examples = read_jsonl(os.path.join(args.data_dir, f"{args.split}.jsonl"))

    score_fn = make_popularity_score_fn(item_pop)
    metrics = evaluate(score_fn, eval_examples, k=args.k)

    print(json.dumps({"model": "popularity", "split": args.split, **metrics}, indent=2))


if __name__ == "__main__":
    main()