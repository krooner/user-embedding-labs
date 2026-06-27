#!/usr/bin/env python3
"""
google/embeddinggemma-300m으로 아이템 텍스트를 인코딩하여 저장한다:
  <out_dir>/item_embeddings.npy   float32, L2 정규화됨, shape (N, dim)
  <out_dir>/item_ids.json         .npy의 행과 정렬된 list[str]

Phase 0에서 유일한 GPU 단계이며, 학습 없이 추론(inference)만 수행한다.
단일 GPU로 전체 Video_Games 카탈로그를 몇 분 안에 인코딩한다.

EmbeddingGemma 고유 참고 사항:
- 아이템은 문서(DOCUMENT)로 인코딩된다 (encode_document가 올바른 prompt를 적용한다).
- 출력 차원은 768이다; --mrl_dim (Matryoshka)으로 512/256/128로 잘라내어 인덱스를
  줄이고 채점 속도를 높인 뒤 재정규화할 수 있다.
- EmbeddingGemma는 float16을 지원하지 않는다; GPU에서는 bfloat16(또는 float32)을 쓴다.
- 가중치가 gated이므로, Hugging Face에서 모델 라이선스에 동의하고 로그인
  (huggingface-cli login)되어 있어야 한다.
"""

import argparse
import json
import os
import sys

import numpy as np


def read_item_texts(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(r["item_id"])
            texts.append(r["text"])
    return ids, texts


def encode_documents(model, texts, batch_size, normalize):
    """가능하면 encode_document를 사용하고(올바른 검색 prompt), 없으면 document
    prompt 이름을 지정한 encode로 대체한다."""
    kwargs = dict(batch_size=batch_size, convert_to_numpy=True,
                  normalize_embeddings=normalize, show_progress_bar=True)
    if hasattr(model, "encode_document"):
        return model.encode_document(texts, **kwargs)
    return model.encode(texts, prompt_name="document", **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_texts", required=True, help="jsonl from build_item_text.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="google/embeddinggemma-300m")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--mrl_dim", type=int, default=0,
                    help="truncate embeddings to this dim (0 = keep full 768)")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    os.makedirs(args.out_dir, exist_ok=True)
    ids, texts = read_item_texts(args.item_texts)
    print(f"encoding {len(texts):,} items with {args.model}", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    if device == "cuda":
        # float16 is unsupported; bfloat16 is safe and faster than fp32
        model = model.to(torch.bfloat16)

    emb = encode_documents(model, texts, args.batch_size, normalize=(args.mrl_dim == 0))
    emb = np.asarray(emb, dtype=np.float32)

    if args.mrl_dim and args.mrl_dim < emb.shape[1]:
        emb = emb[:, : args.mrl_dim]
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # re-normalize

    np.save(os.path.join(args.out_dir, "item_embeddings.npy"), emb)
    with open(os.path.join(args.out_dir, "item_ids.json"), "w") as fp:
        json.dump(ids, fp)

    print(json.dumps({"n_items": len(ids), "dim": int(emb.shape[1])}, indent=2))


if __name__ == "__main__":
    main()