#!/usr/bin/env python3
"""
Amazon Reviews 2023(단일 카테고리)를 사용자 행동 시퀀스로 전처리하며,
leave-one-out 방식으로 train/valid/test를 분할한다.

입력 : 카테고리 리뷰 파일, 예: Video_Games.jsonl  (또는 .jsonl.gz)
출력 : <out_dir>/
          train.jsonl   사용자당 한 줄: {"user_id", "history", "target"}
          valid.jsonl   동일 (target = 뒤에서 두 번째 아이템)
          test.jsonl    동일 (target = 마지막 아이템)
          item_pop.json  {item_id: train_frequency}  -- 인기도 기준선(baseline)용
          stats.json     필터링 이후의 데이터셋 통계

설계 메모 (이렇게 구성한 이유):
- 3GB짜리 jsonl이 한 번에 RAM에 올라가지 않도록 파일을 한 줄씩 스트리밍한다.
  사용자별로 (timestamp, item) 튜플만 유지하므로 메모리 사용량이 작다
  (약 500만 개 상호작용 기준 수백 MB 수준).
- k-core 필터링은 반복적(ITERATIVE)이다: 저빈도 아이템을 제거하면 어떤 사용자가
  임계값 아래로 떨어질 수 있고, 그러면 다시 더 많은 아이템이 임계값 아래로
  내려갈 수 있다. 따라서 안정될 때까지 반복한다.
- 분할은 사용자별로 timestamp 순서에 따른 leave-one-out 방식이다:
      target(test)  = 마지막 아이템
      target(valid) = 뒤에서 두 번째 아이템
      train history = 마지막 두 아이템을 제외한 전부
  인기도는 오직 train history에서만 계산하므로, valid/test가 기준선으로
  새어 들어가는(leakage) 일이 없다.
"""

import argparse
import gzip
import io
import json
import os
import sys
from collections import Counter, defaultdict


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_interactions(path, min_rating=None, verified_only=False, log_every=1_000_000):
    """리뷰 jsonl에서 (user_id, parent_asin, timestamp)를 생성(yield)한다.

    아이템 id로는 asin이 아닌 parent_asin을 사용한다: 데이터셋 문서에 따르면
    색상/크기 변형은 parent_asin을 공유하며, 아이템 메타데이터도 parent_asin을
    키로 한다.
    """
    n_read = 0
    n_kept = 0
    with open_maybe_gzip(path) as fp:
        for line in fp:
            n_read += 1
            if log_every and n_read % log_every == 0:
                print(f"  ...read {n_read:,} lines, kept {n_kept:,}", file=sys.stderr)
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = r.get("user_id")
            item = r.get("parent_asin") or r.get("asin")
            # the field table calls it "timestamp"; some dumps use "sort_timestamp"
            ts = r.get("timestamp", r.get("sort_timestamp"))
            if uid is None or item is None or ts is None:
                continue
            if verified_only and not r.get("verified_purchase", False):
                continue
            if min_rating is not None and (r.get("rating") or 0) < min_rating:
                continue
            n_kept += 1
            yield uid, item, int(ts)
    print(f"  done: read {n_read:,} lines, kept {n_kept:,} interactions", file=sys.stderr)


def build_user_sequences(path, dedup=True, **stream_kwargs):
    """시간 순으로 정렬된 {user_id: [(timestamp, item), ...]}를 반환한다."""
    users = defaultdict(list)
    for uid, item, ts in stream_interactions(path, **stream_kwargs):
        users[uid].append((ts, item))

    for uid, pairs in users.items():
        pairs.sort(key=lambda x: x[0])  # chronological
        if dedup:
            # keep first chronological occurrence of each item
            seen = set()
            deduped = []
            for ts, item in pairs:
                if item in seen:
                    continue
                seen.add(item)
                deduped.append((ts, item))
            users[uid] = deduped
    return users


def kcore_filter(users, k=5, max_iter=50):
    """안정될 때까지 상호작용이 k개 미만인 사용자/아이템을 반복적으로 제거한다."""
    for it in range(max_iter):
        # count item frequencies across all users
        item_counts = Counter()
        for pairs in users.values():
            for _, item in pairs:
                item_counts[item] += 1

        # drop rare items from each user's sequence
        for uid in list(users.keys()):
            users[uid] = [(ts, i) for (ts, i) in users[uid] if item_counts[i] >= k]

        # drop users that are now too short
        before = len(users)
        total_before = sum(len(v) for v in users.values())
        users = {u: v for u, v in users.items() if len(v) >= k}
        after = len(users)
        total_after = sum(len(v) for v in users.values())

        print(f"  k-core iter {it}: users {before}->{after}, "
              f"interactions {total_before}->{total_after}", file=sys.stderr)
        if before == after and total_before == total_after:
            break
    return users


def leave_one_out(users, min_len=3):
    """(split, user_id, history, target) 예시를 생성(yield)한다.

    최소 min_len개의 아이템이 필요하다: train(>=1) + valid(1) + test(1).
    5-core를 적용하면 남은 사용자는 이미 모두 5개 이상을 가지므로, 이는 단순한
    안전장치(guard)일 뿐이다.
    """
    train, valid, test = [], [], []
    item_pop = Counter()
    for uid, pairs in users.items():
        items = [i for _, i in pairs]
        if len(items) < min_len:
            continue
        train_hist = items[:-2]
        valid_hist = items[:-2]
        test_hist = items[:-1]
        valid_target = items[-2]
        test_target = items[-1]

        for i in train_hist:
            item_pop[i] += 1  # popularity from TRAIN portion only -> no leakage

        train.append({"user_id": uid, "history": train_hist[:-1],
                      "target": train_hist[-1] if train_hist else None})
        valid.append({"user_id": uid, "history": valid_hist, "target": valid_target})
        test.append({"user_id": uid, "history": test_hist, "target": test_target})
    return train, valid, test, item_pop


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to <Category>.jsonl(.gz)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k_core", type=int, default=5)
    ap.add_argument("--min_rating", type=float, default=None,
                    help="if set, drop interactions below this rating")
    ap.add_argument("--verified_only", action="store_true")
    ap.add_argument("--no_dedup", action="store_true",
                    help="keep repeated items instead of deduping per user")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] building user sequences (streaming)...", file=sys.stderr)
    users = build_user_sequences(
        args.input,
        dedup=not args.no_dedup,
        min_rating=args.min_rating,
        verified_only=args.verified_only,
    )
    print(f"  raw users: {len(users):,}", file=sys.stderr)

    print(f"[2/4] {args.k_core}-core filtering...", file=sys.stderr)
    users = kcore_filter(users, k=args.k_core)

    print("[3/4] leave-one-out split...", file=sys.stderr)
    train, valid, test, item_pop = leave_one_out(users)

    print("[4/4] writing outputs...", file=sys.stderr)
    write_jsonl(train, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(valid, os.path.join(args.out_dir, "valid.jsonl"))
    write_jsonl(test, os.path.join(args.out_dir, "test.jsonl"))
    with open(os.path.join(args.out_dir, "item_pop.json"), "w") as fp:
        json.dump(item_pop, fp)

    stats = {
        "n_users": len(users),
        "n_items": len(item_pop),
        "n_interactions": sum(len(v) for v in users.values()),
        "avg_seq_len": round(sum(len(v) for v in users.values()) / max(len(users), 1), 2),
        "k_core": args.k_core,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as fp:
        json.dump(stats, fp, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()