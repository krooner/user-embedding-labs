#!/usr/bin/env python3
"""
텍스트 임베딩 기준선(baseline): 각 아이템을 embeddinggemma 벡터로 표현하고,
사용자의 history를 평균 풀링(mean-pool)하여 사용자 벡터로 만든 뒤, 코사인
유사도로 아이템을 랭킹한다. 인기도 기준선과 동일한(SAME) evaluate() 하니스로,
동일한(SAME) split에서 평가한다 -> 비교가 동등(apples-to-apples)하다.

이 파일이 전제하는 파이프라인:
  preprocess.py        -> data_dir/{train,valid,test}.jsonl, item_pop.json
  build_item_text.py   -> item_texts.jsonl
  embed_items.py       -> emb_dir/{item_embeddings.npy, item_ids.json}

설계 선택: 사용자와 아이템은 동일한(SAME) (문서) 공간에 존재한다. 사용자 벡터는
사용자가 상호작용한 아이템들의 문서 임베딩을 L2 정규화한 평균이므로,
cosine(user, candidate_item)은 의미적 유사도 점수가 된다. (이후 변형에서는 대신
텍스트 형태의 사용자 프로필을 구성해 encode_query를 사용할 수도 있다.)
"""

import argparse
import json
import os

import numpy as np

# reuse the harness so both baselines are scored identically
from baseline_popularity import read_jsonl, evaluate


def make_text_embed_score_fn(emb_dir, k):
    E = np.load(os.path.join(emb_dir, "item_embeddings.npy")).astype(np.float32)
    with open(os.path.join(emb_dir, "item_ids.json")) as fp:
        item_ids = json.load(fp)
    id2idx = {pid: i for i, pid in enumerate(item_ids)}
    item_ids_arr = np.array(item_ids, dtype=object)

    def score_fn(history):
        idxs = [id2idx[i] for i in history if i in id2idx]
        if not idxs:
            return []  # no usable history -> counts as a miss (fair)
        uv = E[idxs].mean(axis=0)
        uv /= (np.linalg.norm(uv) + 1e-12)
        scores = E @ uv  # cosine, since rows of E are normalized

        # return enough candidates that, after excluding the user's seen items,
        # the harness still has >= k to choose from. argpartition is O(N).
        n_take = min(len(scores), k + len(idxs) + 5)
        top = np.argpartition(-scores, n_take - 1)[:n_take]
        top = top[np.argsort(-scores[top])]  # sort just the small top slice
        return list(zip(item_ids_arr[top].tolist(), scores[top].tolist()))

    return score_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir from preprocess.py")
    ap.add_argument("--emb_dir", required=True, help="dir from embed_items.py")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    eval_examples = read_jsonl(os.path.join(args.data_dir, f"{args.split}.jsonl"))
    score_fn = make_text_embed_score_fn(args.emb_dir, k=args.k)
    metrics = evaluate(score_fn, eval_examples, k=args.k)

    print(json.dumps(
        {"model": "embeddinggemma-meanpool", "split": args.split, **metrics},
        indent=2))


if __name__ == "__main__":
    main()