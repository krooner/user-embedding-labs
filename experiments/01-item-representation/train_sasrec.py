#!/usr/bin/env python3
"""
SASRec를 학습하고 평가한다. 이 실험의 목적은 모델 성능 자체가 아니라 다음 질문에
대한 통제된(controlled) 비교다:

  백본(SASRec)·학습 목적·평가를 모두 고정했을 때,
  아이템을 'ID 임베딩'으로 표현하는 것과 '텍스트 임베딩'으로 표현하는 것 중
  어느 쪽이 개인화(personalized) next-item 추천에 더 적합한가?

--item_repr {id, text, hybrid} 하나만 바꿔 동일한 코드 경로로 두(또는 세) arm을
학습/평가한다. 입력 lookup과 출력 scoring이 동일한 표현 행렬을 공유한다(weight
tying)는 점이 핵심: 그래서 '표현'이 단일 실험 변수가 된다.

평가 방법론은 popularity / text-meanpool 기준선과 100% 동일하다:
- 같은 split 파일({valid,test}.jsonl), 같은 leave-one-out 타깃.
- 전체 아이템에 대한 full ranking, 사용자가 이미 본 아이템은 제외.
- Recall@k / NDCG@k의 정의는 baseline_popularity.evaluate()와 동일.
배치 평가(evaluate_batched)는 속도를 위한 것이며, --verify_harness 로 기존
evaluate() 하니스와 수치가 일치함을 교차 검증할 수 있다.

추가로 아이템을 train 인기도 구간(cold/warm)으로 나눠 Recall을 분해한다. 표현의
진짜 차이는 보통 롱테일/콜드 아이템에서 드러나기 때문이다.

전제 파이프라인:
  preprocess.py      -> data_dir/{train,valid,test}.jsonl, item_pop.json
  build_item_text.py -> item_texts.jsonl   (text/hybrid arm)
  embed_items.py     -> emb_dir/{item_embeddings.npy, item_ids.json}  (text/hybrid arm)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

# 기준선과 동일한 평가 하니스를 재사용한다 (교차 검증용).
from baseline_popularity import read_jsonl, evaluate
from sasrec import ItemRepresentation, SASRec


# --------------------------------------------------------------------------- #
# 데이터 준비
# --------------------------------------------------------------------------- #
def build_vocab(data_dir):
    """item_pop.json(= train에 등장한 아이템)으로 어휘를 만든다.

    이는 popularity 기준선이 랭킹하는 후보 집합과 정확히 동일하다. train에 한 번도
    안 나온 아이템은 어떤 모델도 표현/랭킹할 수 없으므로(콜드), 모든 방법에서
    동일하게 강제 miss 처리되어 분모가 같아진다.
    """
    with open(os.path.join(data_dir, "item_pop.json")) as fp:
        item_pop = json.load(fp)
    # 인기도 내림차순으로 인덱스를 부여(가독성 목적, 성능과 무관). idx 0 = pad.
    items = sorted(item_pop.items(), key=lambda kv: (-kv[1], kv[0]))
    item2idx = {pid: i + 1 for i, (pid, _) in enumerate(items)}
    pop_by_idx = np.zeros(len(item2idx) + 1, dtype=np.int64)
    for pid, idx in item2idx.items():
        pop_by_idx[idx] = item_pop[pid]
    return item2idx, pop_by_idx


def load_text_matrix(emb_dir, item2idx):
    """어휘 인덱스에 정렬된 텍스트 임베딩 행렬 (V+1, d_text)을 만든다.

    행 0 = padding(0). emb에 없는 아이템(거의 없음)은 0 벡터로 두고 개수를 보고한다.
    """
    E = np.load(os.path.join(emb_dir, "item_embeddings.npy")).astype(np.float32)
    with open(os.path.join(emb_dir, "item_ids.json")) as fp:
        item_ids = json.load(fp)
    pid2row = {pid: i for i, pid in enumerate(item_ids)}

    d_text = E.shape[1]
    V = len(item2idx)
    mat = np.zeros((V + 1, d_text), dtype=np.float32)
    n_missing = 0
    for pid, idx in item2idx.items():
        row = pid2row.get(pid)
        if row is None:
            n_missing += 1
            continue
        mat[idx] = E[row]
    if n_missing:
        print(f"  [warn] {n_missing} vocab items lack a text embedding "
              f"(zero vector used)", file=sys.stderr)
    return mat


def pad_seq(seq_idx, maxlen):
    """정수 인덱스 리스트를 길이 maxlen으로 오른쪽 패딩(뒤쪽 0 채움).

    가장 최근 maxlen개를 유지하고(앞쪽 절단), 뒤쪽을 0으로 채운다. right-pad이므로
    위치 0은 항상 실제 아이템 -> causal+padding mask 조합에서 NaN이 생기지 않는다.
    """
    seq = seq_idx[-maxlen:]
    return seq + [0] * (maxlen - len(seq))


def build_train_arrays(data_dir, item2idx, maxlen):
    """train 사용자 시퀀스로부터 (input, label) 배열을 만든다.

    train 시퀀스 = history + [target] (= 원 시퀀스의 items[:-2], 즉 train 구간).
    autoregressive: input = s[:-1], label = s[1:] (다음 아이템 예측).
    둘 다 동일하게 left-pad. label 0(pad)은 손실에서 무시된다.
    """
    rows = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    inputs, labels = [], []
    for ex in rows:
        seq_pids = list(ex["history"])
        if ex.get("target") is not None:
            seq_pids.append(ex["target"])
        s = [item2idx[p] for p in seq_pids if p in item2idx]
        if len(s) < 2:
            continue
        inp = s[:-1]
        lab = s[1:]
        inputs.append(pad_seq(inp, maxlen))
        labels.append(pad_seq(lab, maxlen))
    return (np.asarray(inputs, dtype=np.int64),
            np.asarray(labels, dtype=np.int64))


def build_eval_arrays(data_dir, split, item2idx, maxlen):
    """평가 split의 (history_input, target_idx) 배열을 만든다.

    target이 어휘에 없으면(콜드) target_idx = 0 -> 절대 맞힐 수 없는 강제 miss.
    이는 popularity 기준선의 동작과 동일하다(분모 동일).
    """
    rows = read_jsonl(os.path.join(data_dir, f"{split}.jsonl"))
    hist, targets = [], []
    for ex in rows:
        h = [item2idx[p] for p in ex["history"] if p in item2idx]
        hist.append(pad_seq(h, maxlen))
        t = ex["target"]
        targets.append(item2idx.get(t, 0) if t is not None else 0)
    return (np.asarray(hist, dtype=np.int64),
            np.asarray(targets, dtype=np.int64))


# --------------------------------------------------------------------------- #
# 평가 (배치) — baseline_popularity.evaluate() 와 동일한 지표 정의
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_batched(model, hist_arr, target_arr, pop_by_idx, k=10,
                     batch_size=512, device="cuda", pop_buckets=None):
    """full ranking, seen 제외, Recall@k / NDCG@k.

    하니스 evaluate()와 동등함: 점수에서 (이미 본 아이템 + pad)을 -inf로 만든 뒤
    상위 k개를 고르는 것은, evaluate()가 정렬 리스트에서 seen을 건너뛰며 k개를
    모으는 것과 정확히 같은 top-k 집합/순서를 만든다.

    pop_buckets: [(name, lo, hi), ...] -- 타깃의 train 인기도가 [lo, hi)이면 그
    버킷에 카운트하여 구간별 Recall@k를 분해한다.
    """
    model.eval()
    M = model.item_repr.matrix().detach().to(device)  # (V+1, d)
    n = hist_arr.shape[0]

    recall_sum = 0.0
    ndcg_sum = 0.0
    # 인기도 구간별 분해
    buckets = pop_buckets or []
    b_hit = [0.0] * len(buckets)
    b_tot = [0] * len(buckets)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        seq = torch.as_tensor(hist_arr[start:end], device=device)      # (B, L)
        tgt = torch.as_tensor(target_arr[start:end], device=device)    # (B,)

        uv = model.user_vector(seq)            # (B, d)
        scores = uv @ M.t()                    # (B, V+1)
        # 이미 본 아이템 + pad(열 0) 제외: 입력에 등장한 인덱스를 -inf로.
        scores.scatter_(1, seq, float("-inf"))
        scores[:, 0] = float("-inf")

        topk = torch.topk(scores, k, dim=1).indices    # (B, k)
        match = topk == tgt.unsqueeze(1)                # (B, k)
        hit = match.any(dim=1)                          # (B,)
        # 사용할 history가 없는 사용자는 강제 miss (score_fn의 [] 반환과 동일).
        nonempty = (seq != 0).any(dim=1)
        hit = hit & nonempty
        # 1-기반 순위 (맞힌 경우에만 의미 있음)
        rank = match.float().argmax(dim=1) + 1
        gain = torch.where(hit, 1.0 / torch.log2(rank.float() + 1),
                           torch.zeros_like(rank, dtype=torch.float))

        # 타깃이 어휘 밖(idx 0)이면 강제 miss (topk엔 0이 없음) -> 자동 처리됨.
        recall_sum += hit.float().sum().item()
        ndcg_sum += gain.sum().item()

        if buckets:
            tgt_pop = torch.as_tensor(pop_by_idx[target_arr[start:end]],
                                      device=device)
            for bi, (_, lo, hi) in enumerate(buckets):
                in_b = (tgt_pop >= lo) & (tgt_pop < hi) & (tgt > 0)
                b_tot[bi] += int(in_b.sum().item())
                b_hit[bi] += float((hit & in_b).float().sum().item())

    out = {
        "n_eval": n,
        f"Recall@{k}": round(recall_sum / n, 4) if n else 0.0,
        f"NDCG@{k}": round(ndcg_sum / n, 4) if n else 0.0,
    }
    if buckets:
        out["by_popularity"] = {
            buckets[bi][0]: {
                "n": b_tot[bi],
                f"Recall@{k}": round(b_hit[bi] / b_tot[bi], 4) if b_tot[bi] else None,
            }
            for bi in range(len(buckets))
        }
    return out


def make_model_score_fn(model, item2idx, idx2pid, maxlen, k, device):
    """학습된 모델을 baseline_popularity.evaluate() 와 호환되는 score_fn으로 감싼다.

    이로써 SASRec를 popularity / text-meanpool 과 '같은 함수'로 채점해 교차 검증할
    수 있다 (text 기준선과 동일한 argpartition top-N 반환 패턴).
    """
    M = model.item_repr.matrix().detach().to(device)

    @torch.no_grad()
    def score_fn(history):
        idxs = [item2idx[p] for p in history if p in item2idx]
        if not idxs:
            return []  # 사용할 history 없음 -> miss (text 기준선과 동일)
        seq = torch.as_tensor([pad_seq(idxs, maxlen)], device=device)
        uv = model.user_vector(seq)              # (1, d)
        scores = (uv @ M.t()).squeeze(0)         # (V+1,)
        scores[0] = float("-inf")
        n_take = min(scores.shape[0], k + len(idxs) + 5)
        top = torch.topk(scores, n_take).indices.tolist()
        return [(idx2pid[i], scores[i].item()) for i in top if i != 0]

    return score_fn


# --------------------------------------------------------------------------- #
# 학습
# --------------------------------------------------------------------------- #
def train(model, train_in, train_lab, device, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # label 0(pad) 무시
    n = train_in.shape[0]
    train_in_t = torch.as_tensor(train_in)
    train_lab_t = torch.as_tensor(train_lab)

    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        total = 0.0
        nb = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            seq = train_in_t[idx].to(device)        # (B, L)
            lab = train_lab_t[idx].to(device)       # (B, L)

            h = model.seq_hidden(seq)               # (B, L, d)
            valid = lab != 0                        # 손실 대상 위치
            if valid.sum() == 0:
                continue
            hv = h[valid]                           # (P, d)
            labv = lab[valid]                       # (P,)
            M = model.item_repr.matrix()            # (V+1, d), weight tying
            logits = hv @ M.t()                     # (P, V+1)
            loss = loss_fn(logits, labv)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        yield epoch, total / max(nb, 1)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="preprocess.py 산출 디렉토리")
    ap.add_argument("--emb_dir", default=None,
                    help="embed_items.py 산출 디렉토리 (text/hybrid arm 필수)")
    ap.add_argument("--item_repr", default="id",
                    choices=["id", "text", "hybrid", "randproj"])
    ap.add_argument("--train_text", action="store_true",
                    help="text 임베딩을 frozen이 아닌 학습 대상으로 (비쌈)")
    ap.add_argument("--latent_dim", type=int, default=768,
                    help="randproj 대조군의 무작위 테이블 차원 (text(ft)와 맞춤)")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5,
                    help="valid Recall@k 가 개선되지 않을 때 조기 종료 인내 횟수")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--verify_harness", type=int, default=0,
                    help="첫 N명을 기존 evaluate() 하니스로 교차 검증 (0=생략)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로(선택)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, item_repr={args.item_repr}", file=sys.stderr)

    # --- 데이터 ---
    item2idx, pop_by_idx = build_vocab(args.data_dir)
    idx2pid = {i: p for p, i in item2idx.items()}
    V = len(item2idx)
    print(f"vocab(train items) V={V:,}", file=sys.stderr)

    text_matrix = None
    if args.item_repr in ("text", "hybrid"):
        assert args.emb_dir, "text/hybrid arm 에는 --emb_dir 가 필요하다"
        text_matrix = load_text_matrix(args.emb_dir, item2idx)

    print("building train/eval arrays...", file=sys.stderr)
    train_in, train_lab = build_train_arrays(args.data_dir, item2idx, args.maxlen)
    valid_h, valid_t = build_eval_arrays(args.data_dir, "valid", item2idx, args.maxlen)
    test_h, test_t = build_eval_arrays(args.data_dir, "test", item2idx, args.maxlen)
    print(f"train sequences: {train_in.shape[0]:,}", file=sys.stderr)

    # 인기도 구간 (train frequency 기준). 5-core 이므로 최소 5.
    pop_buckets = [("cold[5,20)", 5, 20), ("mid[20,100)", 20, 100),
                   ("hot[100,inf)", 100, 10**9)]

    # --- 모델 ---
    item_repr = ItemRepresentation(
        args.item_repr, V, args.d_model, text_matrix=text_matrix,
        train_text=args.train_text, latent_dim=args.latent_dim,
        dropout=args.dropout)
    model = SASRec(item_repr, args.d_model, args.maxlen,
                   n_layers=args.n_layers, n_heads=args.n_heads,
                   dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}", file=sys.stderr)

    # --- 학습 + 조기 종료 ---
    best_recall = -1.0
    best_state = None
    bad = 0
    rk = f"Recall@{args.k}"
    for epoch, loss in train(model, train_in, train_lab, device, args):
        msg = f"epoch {epoch:3d}  loss {loss:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            vm = evaluate_batched(model, valid_h, valid_t, pop_by_idx,
                                  k=args.k, device=device)
            msg += f"  valid {rk} {vm[rk]:.4f}  NDCG@{args.k} {vm[f'NDCG@{args.k}']:.4f}"
            if vm[rk] > best_recall:
                best_recall = vm[rk]
                best_state = {k_: v.detach().cpu().clone()
                              for k_, v in model.state_dict().items()}
                bad = 0
                msg += "  *"
            else:
                bad += 1
        print(msg, file=sys.stderr)
        if bad >= args.patience:
            print(f"early stop (no valid improvement for {args.patience} evals)",
                  file=sys.stderr)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 최종 test 평가 (인기도 구간 분해 포함) ---
    test_metrics = evaluate_batched(model, test_h, test_t, pop_by_idx,
                                    k=args.k, device=device, pop_buckets=pop_buckets)

    # --- 선택: 기존 하니스로 교차 검증 ---
    if args.verify_harness > 0:
        rows = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))[:args.verify_harness]
        score_fn = make_model_score_fn(model, item2idx, idx2pid,
                                       args.maxlen, args.k, device)
        hm = evaluate(score_fn, rows, k=args.k)
        bm = evaluate_batched(model, test_h[:args.verify_harness],
                              test_t[:args.verify_harness], pop_by_idx,
                              k=args.k, device=device)
        print(f"[verify] first {args.verify_harness} users  "
              f"harness {rk}={hm[rk]}  batched {rk}={bm[rk]}", file=sys.stderr)

    result = {
        "model": f"sasrec-{args.item_repr}",
        "split": "test",
        "d_model": args.d_model,
        "trainable_params": n_params,
        "best_valid_recall": round(best_recall, 4),
        **test_metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
