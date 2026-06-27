#!/usr/bin/env python3
"""
Zero-shot(콜드 아이템) 평가: 텍스트 표현이 '학습 중 한 번도 본 적 없는' 아이템을
그 텍스트만으로 추천할 수 있는가? ID 표현은 구조적으로 불가능하다.

프로토콜 (simulated holdout):
- train 어휘에서 무작위로 일부(--holdout_frac)를 COLD 집합으로 지정한다.
- COLD 아이템을 모든 '학습' 시퀀스에서 제거한다 -> 모델은 이들을 학습하지 못한다.
  (두 arm 모두 동일하게 축소된 학습 데이터를 쓰므로 공정하다.)
- 평가는 test 타깃이 COLD인 사용자에 대해 수행한다. history는 WARM 아이템만 남긴다.
  후보(catalog)는 WARM ∪ COLD 전체.
    * ID arm   : COLD 아이템 표현이 없음 -> 후보는 WARM뿐 -> COLD 타깃 Recall = 0 (구조적).
    * text arm : 학습된 projection을 COLD 아이템의 frozen embeddinggemma 벡터에 적용해
                 표현 -> COLD 타깃을 랭킹할 수 있다. 이것이 zero-shot 일반화의 핵심.

참고로 같은 모델의 WARM 타깃 성능도 함께 보고하여, cold가 warm 대비 얼마나 회복되는지
맥락을 제공한다. text(ft)(per-item 학습 테이블)는 COLD 행이 없어 zero-shot에 부적합하므로
여기서는 frozen text 표현을 사용한다(텍스트→공간 사상이 모든 아이템에 적용 가능).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from baseline_popularity import read_jsonl
from sasrec import ItemRepresentation, SASRec
import train_sasrec as T


def train_arm(model, train_in, train_lab, valid_h, valid_t, pop_by_idx,
              device, args, tag):
    """early stopping 학습. train_sasrec.main()의 루프와 동일."""
    best, best_state, bad = -1.0, None, 0
    rk = f"Recall@{args.k}"
    for epoch, loss in T.train(model, train_in, train_lab, device, args):
        vm = T.evaluate_batched(model, valid_h, valid_t, pop_by_idx,
                                k=args.k, device=device)
        improved = vm[rk] > best
        if improved:
            best, bad = vm[rk], 0
            best_state = {kk: v.detach().cpu().clone()
                          for kk, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  [{tag}] epoch {epoch:3d} loss {loss:.4f} valid {rk} {vm[rk]:.4f}"
              f"{'  *' if improved else ''}", file=sys.stderr)
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


@torch.no_grad()
def eval_over_catalog(model, M, eval_pids, pid2evalidx, target_set,
                      warm2idx, test_rows, maxlen, k, device,
                      bias=None, normalize=False):
    """확장 catalog(M: (N,d))에 대해 target이 target_set에 속한 사용자만 평가한다.

    history는 warm2idx로 매핑(=WARM만), 이미 본 아이템은 후보에서 제외, full ranking.
    지표 정의는 baseline_popularity.evaluate()와 동일.

    비대칭 scoring 지원 (실험 8):
      normalize=True -> uv·M을 cosine으로 (uv와 M 행을 모두 단위 노름). bias 항과
        섞을 때 cosine을 [-1,1]로 고정해 β를 해석 가능하게 만든다.
      bias (N,) -> 점수에 더하는 per-item 항(예: β·log(1+train_pop)). cold는 train_pop=0
        이라 bias=0이 자연스럽게 성립 -> warm에만 인기도 prior가 가산된다.
    """
    if normalize:
        M = M / (M.norm(dim=1, keepdim=True) + 1e-12)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.as_tensor(bias, dtype=torch.float32, device=device)
    recall = ndcg = 0.0
    n = 0
    for ex in test_rows:
        tgt = ex["target"]
        if tgt not in target_set or tgt not in pid2evalidx:
            continue
        hist = [warm2idx[p] for p in ex["history"] if p in warm2idx]
        if not hist:
            continue
        seq = torch.as_tensor([T.pad_seq(hist, maxlen)], device=device)
        uv = model.user_vector(seq)                 # (1, d)
        if normalize:
            uv = uv / (uv.norm(dim=1, keepdim=True) + 1e-12)
        scores = (uv @ M.t()).squeeze(0).clone()    # (N,)
        if bias is not None:
            scores = scores + bias
        # 이미 본 WARM 아이템 제외 (catalog 내 index = warm2idx-1)
        seen = [warm2idx[p] - 1 for p in ex["history"] if p in warm2idx]
        scores[seen] = float("-inf")
        topk = torch.topk(scores, k).indices.tolist()
        n += 1
        ti = pid2evalidx[tgt]
        if ti in topk:
            rank = topk.index(ti) + 1
            recall += 1.0
            ndcg += 1.0 / math.log2(rank + 1)
    return {"n_eval": n,
            f"Recall@{k}": round(recall / n, 4) if n else 0.0,
            f"NDCG@{k}": round(ndcg / n, 4) if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.10,
                    help="COLD(미학습)로 보류할 train 어휘 비율")
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_sample", type=int, default=5000,
                    help="warm 타깃 참고 평가에 쓸 사용자 표본 수 (속도)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 어휘 + COLD holdout ---
    with open(os.path.join(args.data_dir, "item_pop.json")) as fp:
        item_pop = json.load(fp)
    all_items = sorted(item_pop.keys())
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_items))
    n_hold = int(len(all_items) * args.holdout_frac)
    cold_set = set(all_items[i] for i in perm[:n_hold])
    warm_items = [it for it in all_items if it not in cold_set]
    warm2idx = {pid: i + 1 for i, pid in enumerate(warm_items)}
    Vw = len(warm_items)
    pop_by_idx = np.zeros(Vw + 1, dtype=np.int64)
    for pid, idx in warm2idx.items():
        pop_by_idx[idx] = item_pop[pid]
    print(f"warm={Vw:,}  cold(held-out)={len(cold_set):,}  device={device}",
          file=sys.stderr)

    # --- 임베딩 (warm + cold 모두 필요) ---
    E = np.load(os.path.join(args.emb_dir, "item_embeddings.npy")).astype(np.float32)
    with open(os.path.join(args.emb_dir, "item_ids.json")) as fp:
        item_ids = json.load(fp)
    pid2row = {pid: i for i, pid in enumerate(item_ids)}
    d_text = E.shape[1]
    # WARM 학습용 텍스트 행렬 (Vw+1, d_text), 행 0 = pad
    warm_text = np.zeros((Vw + 1, d_text), dtype=np.float32)
    for pid, idx in warm2idx.items():
        r = pid2row.get(pid)
        if r is not None:
            warm_text[idx] = E[r]

    # --- 학습 데이터 (COLD 자동 제거: warm2idx에 없는 아이템은 build에서 drop) ---
    train_in, train_lab = T.build_train_arrays(args.data_dir, warm2idx, args.maxlen)
    valid_h, valid_t = T.build_eval_arrays(args.data_dir, "valid", warm2idx, args.maxlen)
    print(f"train sequences (cold 제거 후): {train_in.shape[0]:,}", file=sys.stderr)

    # --- 두 arm 학습 (동일 데이터, 표현만 다름) ---
    print("[train] ID arm ...", file=sys.stderr)
    id_repr = ItemRepresentation("id", Vw, args.d_model, dropout=args.dropout)
    id_model = SASRec(id_repr, args.d_model, args.maxlen, args.n_layers,
                      args.n_heads, args.dropout).to(device)
    train_arm(id_model, train_in, train_lab, valid_h, valid_t, pop_by_idx,
              device, args, "id")

    print("[train] text(frozen) arm ...", file=sys.stderr)
    tx_repr = ItemRepresentation("text", Vw, args.d_model, text_matrix=warm_text,
                                 train_text=False, dropout=args.dropout)
    tx_model = SASRec(tx_repr, args.d_model, args.maxlen, args.n_layers,
                      args.n_heads, args.dropout).to(device)
    train_arm(tx_model, train_in, train_lab, valid_h, valid_t, pop_by_idx,
              device, args, "text")

    print("[train] hybrid(frozen) arm ...", file=sys.stderr)
    # hybrid: 학습되는 ID 테이블(WARM 전용) + frozen 텍스트 prior. text와 동일하게
    # frozen이어야 학습된 projection을 미학습 COLD 아이템에도 적용할 수 있다.
    hy_repr = ItemRepresentation("hybrid", Vw, args.d_model, text_matrix=warm_text,
                                 train_text=False, dropout=args.dropout)
    hy_model = SASRec(hy_repr, args.d_model, args.maxlen, args.n_layers,
                      args.n_heads, args.dropout).to(device)
    train_arm(hy_model, train_in, train_lab, valid_h, valid_t, pop_by_idx,
              device, args, "hybrid")

    # --- 평가 catalog 구성 ---
    cold_items = sorted(cold_set)
    test_rows = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    warm_set_pre = set(warm_items)
    cold_rows = [r for r in test_rows if r["target"] in cold_set]          # 전부
    warm_ref_rows = [r for r in test_rows
                     if r["target"] in warm_set_pre][:args.ref_sample]     # 표본
    print(f"eval users: cold-target={len(cold_rows):,}  "
          f"warm-ref(sample)={len(warm_ref_rows):,}", file=sys.stderr)

    # ID catalog = WARM만 (cold 표현 없음). M_id = id 테이블의 행 1..Vw.
    eval_pids_id = warm_items
    pid2evalidx_id = {p: i for i, p in enumerate(eval_pids_id)}
    M_id = id_model.item_repr.matrix().detach()[1:].to(device)  # (Vw, d)

    # text catalog = WARM ∪ COLD. 학습된 projection을 frozen 텍스트 벡터에 적용.
    eval_pids_tx = warm_items + cold_items
    pid2evalidx_tx = {p: i for i, p in enumerate(eval_pids_tx)}
    E_eval = np.stack([E[pid2row[p]] for p in eval_pids_tx]).astype(np.float32)
    ir = tx_model.item_repr
    with torch.no_grad():
        M_tx = ir.text_norm(ir.text_proj(torch.as_tensor(E_eval, device=device)))

    # hybrid catalog = WARM ∪ COLD. WARM은 id_emb + proj(text)(=matrix()의 학습된 행),
    # COLD는 id 행이 없으므로 proj(text)만 적용 -> 자연스러운 zero-shot fallback.
    eval_pids_hy = warm_items + cold_items                       # text와 동일 순서
    pid2evalidx_hy = {p: i for i, p in enumerate(eval_pids_hy)}
    hir = hy_model.item_repr
    E_cold = np.stack([E[pid2row[p]] for p in cold_items]).astype(np.float32)
    with torch.no_grad():
        M_hy_warm = hir.matrix().detach()[1:].to(device)         # (Vw, d): id+proj(text)
        M_hy_cold = hir.text_norm(hir.text_proj(                 # (n_cold, d): proj(text)만
            torch.as_tensor(E_cold, device=device)))
        M_hy = torch.cat([M_hy_warm, M_hy_cold], dim=0)

    # L2 정규화 변형: catalog 각 행을 단위 노름으로 -> item쪽 cosine scoring.
    # hybrid의 warm(id+text, 노름 큼) vs cold(text만, 노름 작음) 비대칭을 제거해
    # cold가 랭킹에서 공정하게 경쟁하는지 검증. (user 벡터 정규화는 argsort 불변이라 생략)
    def l2norm(M):
        return M / (M.norm(dim=1, keepdim=True) + 1e-12)
    M_hy_l2 = l2norm(M_hy)
    M_tx_l2 = l2norm(M_tx)

    warm_set = set(warm_items)

    # --- 실험 8: 비대칭 scoring = cosine + β·log(1+train_pop) ---
    # cosine(=normalize=True)은 warm/cold를 방향으로 공정하게 비교(cold의 강점)하고,
    # popularity bias는 L2가 버린 warm의 노름 신호를 명시적으로 복원한다. cold는 학습에서
    # held-out -> train_pop=0 -> bias=0 자연 성립 -> warm에만 인기도 prior가 가산된다.
    # β=0이면 순수 cosine(=실험 7의 hybrid_frozen_l2)이라 내부 검증도 된다.
    pop_cat = np.array([float(item_pop[p]) for p in warm_items]
                       + [0.0] * len(cold_items), dtype=np.float32)
    logpop = np.log1p(pop_cat)                                   # (N,)
    logpop = logpop / (logpop.max() + 1e-12)   # [0,1]로 정규화 -> β를 cosine 단위로
    betas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    asym = {}
    for b in betas:
        bias = b * logpop
        asym[f"beta={b}"] = {
            "cold": eval_over_catalog(
                hy_model, M_hy, eval_pids_hy, pid2evalidx_hy, cold_set, warm2idx,
                cold_rows, args.maxlen, args.k, device, bias=bias, normalize=True),
            "warm": eval_over_catalog(
                hy_model, M_hy, eval_pids_hy, pid2evalidx_hy, warm_set, warm2idx,
                warm_ref_rows, args.maxlen, args.k, device, bias=bias, normalize=True),
        }
        print(f"[asym] β={b}: cold {asym[f'beta={b}']['cold']}  "
              f"warm {asym[f'beta={b}']['warm']}", file=sys.stderr)

    # --- 결과 ---
    res = {
        "holdout_frac": args.holdout_frac,
        "warm_items": Vw, "cold_items": len(cold_set),
        "asym_scoring_cosine_plus_beta_logpop": asym,
        "cold_target": {
            # ID: cold 타깃은 WARM catalog에 없음 -> 구조적 0. (참고로 catalog/타깃셋이
            #     겹치지 않아 n_eval은 동일 사용자 수로 맞추기 위해 text 경로로 카운트)
            "ID": {"Recall@{}".format(args.k): 0.0, "NDCG@{}".format(args.k): 0.0,
                   "note": "cold 아이템 표현이 없어 구조적으로 0"},
            "text_frozen": eval_over_catalog(
                tx_model, M_tx, eval_pids_tx, pid2evalidx_tx, cold_set,
                warm2idx, cold_rows, args.maxlen, args.k, device),
            "hybrid_frozen": eval_over_catalog(
                hy_model, M_hy, eval_pids_hy, pid2evalidx_hy, cold_set,
                warm2idx, cold_rows, args.maxlen, args.k, device),
            "hybrid_frozen_l2": eval_over_catalog(
                hy_model, M_hy_l2, eval_pids_hy, pid2evalidx_hy, cold_set,
                warm2idx, cold_rows, args.maxlen, args.k, device),
            "text_frozen_l2": eval_over_catalog(
                tx_model, M_tx_l2, eval_pids_tx, pid2evalidx_tx, cold_set,
                warm2idx, cold_rows, args.maxlen, args.k, device),
        },
        "warm_target_reference": {
            "ID": eval_over_catalog(
                id_model, M_id, eval_pids_id, pid2evalidx_id, warm_set,
                warm2idx, warm_ref_rows, args.maxlen, args.k, device),
            "text_frozen": eval_over_catalog(
                tx_model, M_tx, eval_pids_tx, pid2evalidx_tx, warm_set,
                warm2idx, warm_ref_rows, args.maxlen, args.k, device),
            "hybrid_frozen": eval_over_catalog(
                hy_model, M_hy, eval_pids_hy, pid2evalidx_hy, warm_set,
                warm2idx, warm_ref_rows, args.maxlen, args.k, device),
            "hybrid_frozen_l2": eval_over_catalog(
                hy_model, M_hy_l2, eval_pids_hy, pid2evalidx_hy, warm_set,
                warm2idx, warm_ref_rows, args.maxlen, args.k, device),
            "text_frozen_l2": eval_over_catalog(
                tx_model, M_tx_l2, eval_pids_tx, pid2evalidx_tx, warm_set,
                warm2idx, warm_ref_rows, args.maxlen, args.k, device),
        },
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(res, fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
