#!/usr/bin/env python3
"""
카테고리 메타데이터(METADATA) 파일(예: meta_Video_Games.jsonl)을 읽어,
전처리된 데이터셋의 모든 아이템에 대해 {item_id (parent_asin): text}를 구성한다.

왜 별도 단계인가: preprocess.py는 시퀀스에 아이템 ID만 남긴다. 아이템을 텍스트로
표현하려면 title/description이 필요하며, 이는 메타데이터 파일(parent_asin을 키로
함)에 들어 있다. 우리 split에 실제로 등장하는 아이템에 대해서만 텍스트를 구성하므로,
13.7만 개에 달하는 전체 카탈로그를 불필요하게 인코딩하지 않는다.

아이템 텍스트 = title + features + description (잘라냄). 이것이 임베딩 모델이 보게
될 내용이다. 일부 아이템에는 메타데이터가 없다고 문서에 명시되어 있으므로,
메타데이터 커버리지(coverage)를 함께 보고한다.
"""

import argparse
import gzip
import io
import json
import os
import sys


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def collect_item_universe(data_dir):
    """모든 split에 걸친 아이템들의 합집합. test.jsonl의 history+target만으로도 이미
    각 사용자의 전체 시퀀스가 포함되지만, 안전을 위해 세 파일을 모두 읽는다."""
    universe = set()
    for split in ["train", "valid", "test"]:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                ex = json.loads(line)
                universe.update(ex.get("history", []))
                if ex.get("target") is not None:
                    universe.add(ex["target"])
    return universe


def as_text(meta, max_chars=1200):
    """아이템 메타데이터 필드들로부터 하나의 텍스트 문자열을 구성한다."""
    parts = []
    if meta.get("title"):
        parts.append(str(meta["title"]))
    feats = meta.get("features") or []
    if feats:
        parts.append(" ".join(map(str, feats)))
    desc = meta.get("description") or []
    if desc:
        parts.append(" ".join(map(str, desc)))
    store = meta.get("store")
    if store:
        parts.append(f"Store: {store}")
    text = " ".join(parts).strip()
    return text[:max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="meta_<Category>.jsonl(.gz)")
    ap.add_argument("--data_dir", required=True, help="dir produced by preprocess.py")
    ap.add_argument("--out", required=True, help="output jsonl: {item_id, text}")
    ap.add_argument("--max_chars", type=int, default=1200)
    args = ap.parse_args()

    print("collecting item universe from splits...", file=sys.stderr)
    universe = collect_item_universe(args.data_dir)
    print(f"  {len(universe):,} unique items needed", file=sys.stderr)

    found = {}
    n_read = 0
    with open_maybe_gzip(args.meta) as fp:
        for line in fp:
            n_read += 1
            if n_read % 200_000 == 0:
                print(f"  ...scanned {n_read:,} meta rows, matched {len(found):,}",
                      file=sys.stderr)
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = meta.get("parent_asin")
            if pid is None or pid not in universe or pid in found:
                continue
            text = as_text(meta, args.max_chars)
            if not text:
                text = pid  # fallback so the item is never empty
            found[pid] = text

    # items with no metadata at all -> fall back to the id as text
    missing = universe - set(found.keys())
    for pid in missing:
        found[pid] = pid

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        for pid, text in found.items():
            fp.write(json.dumps({"item_id": pid, "text": text}) + "\n")

    coverage = 1.0 - len(missing) / max(len(universe), 1)
    print(json.dumps({
        "items_needed": len(universe),
        "items_with_metadata": len(universe) - len(missing),
        "metadata_coverage": round(coverage, 4),
    }, indent=2))


if __name__ == "__main__":
    main()