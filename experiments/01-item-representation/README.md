# User Embedding — Phase 0: Item Representation 비교

**핵심 질문:** 추천 모델의 입력 아이템 표현(item representation)으로 단순 **ID**를
쓰는 것과, 메타데이터의 **텍스트 임베딩**(embeddinggemma-300m)을 쓰는 것 중 무엇이
개인화 추천에 더 적합한가?

단일 카테고리(Video_Games)에서, 평가 하니스를 고정하고 표현만 바꿔 가며 일련의
통제된 실험으로 이 질문을 좁혀 나간다. 아래는 그 실험 로그다.

---

## 데이터 파이프라인 (전 실험 공통)
```bash
# 0) 데이터 다운로드 (직접)
# metadata
wget https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Video_Games.jsonl
# review
wget https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Video_Games.jsonl

# 1) 시퀀스 생성 + 5-core + leave-one-out 분할
uv run python3 preprocess.py --input Video_Games.jsonl --out_dir data/video_games --k_core 5

# 2) 메타데이터 -> 아이템 텍스트 (split에 등장하는 아이템만)
uv run python3 build_item_text.py --meta meta_Video_Games.jsonl \
    --data_dir data/video_games --out data/video_games/item_texts.jsonl

# 3) embeddinggemma로 아이템 인코딩 (GPU, 추론 전용)
uv run python3 embed_items.py --item_texts data/video_games/item_texts.jsonl \
    --model $HOME/models/embeddinggemma-300m \
    --out_dir data/video_games/emb --batch_size 256
```

**데이터 규모:** 사용자 94,762 · 아이템 25,527 · 상호작용 814,586 · 평균 시퀀스 8.6 · 5-core

## 공통 평가 프로토콜
- **Leave-one-out**: 사용자당 held-out 타깃 1개. valid=뒤에서 2번째, test=마지막.
- **Full ranking**: 샘플링된 negative가 아닌 전체 아이템 대상 랭킹 (샘플링 메트릭의
  왜곡 회피, Krichene & Rendle 2020). 이미 본 아이템은 후보에서 제외.
- **지표**: Recall@10 = Hit@10(단일 타깃), NDCG@10 = 1/log2(rank+1).
- **누수 방지**: 인기도/학습은 train 구간에서만. 후보 집합은 비교 arm 간 동일하게 고정.
- 모든 방법은 `baseline_popularity.py`의 동일한 `evaluate()`로 채점된다. 바뀌는 것은
  오직 `score_fn`(=표현)뿐 → apples-to-apples.

---

# 실험 0 — 정적 기준선 (학습 없는 표현 비교)

### 1. 가설
학습 없이도, 텍스트 임베딩 표현이 단순 인기도(popularity)보다 더 나은 개인화 추천을
줄 것이다. (사용자 history를 임베딩 평균으로 집계 → cosine 랭킹)

### 2. 실행 코드
```bash
uv run python3 baseline_popularity.py  --data_dir data/video_games --split test
uv run python3 baseline_text_embed.py  --data_dir data/video_games \
    --emb_dir data/video_games/emb --split test
```

### 3. 결과 및 결론
| 방법 | Recall@10 | NDCG@10 |
|---|---|---|
| popularity | 0.0249 | 0.0126 |
| text mean-pool (학습 없음) | 0.0253 | 0.0130 |

거의 동일하다. **가설은 검증 불가(설계 결함)**: mean-pool+cosine은 학습이 없는
비개인화 휴리스틱이라, 표현의 좋고 나쁨이 아니라 *집계 방식*의 한계를 보고 있을 뿐이다.
표현의 가치는 표현 외 요인(집계)에 가려진다.

### 4. 다음 작업 / 요약
표현의 가치를 격리하려면 **백본·학습 목적·평가를 고정하고 표현만 단일 변수로** 바꿔야
한다 → 학습된 시퀀스 모델(SASRec)에 ID/text 표현을 끼워 비교(실험 1).

---

# 실험 1 — 학습된 SASRec: ID vs text(frozen)

### 1. 가설
같은 SASRec 백본에서, 입력 아이템 표현을 ID 대신 frozen 텍스트 임베딩으로 바꾸면
추천 성능이 더 좋아질 것이다.

### 2. 실행 코드
```bash
uv run python3 train_sasrec.py --data_dir data/video_games --item_repr id \
    --epochs 80 --patience 6 --out results/sasrec_id.json
uv run python3 train_sasrec.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --item_repr text --epochs 80 --patience 6 --out results/sasrec_text.json
# --verify_harness N : 배치 평가가 baseline_popularity.evaluate()와 일치함을 교차검증
```
설계: 입력 lookup과 출력 scoring이 **동일 표현 행렬을 공유(weight tying)** → 표현이
유일한 변수. 후보는 양쪽 모두 train 어휘로 고정.

### 3. 결과 및 결론
| 방법 | 차원 | 학습 파라미터 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| SASRec — ID | 64 | 1.69M | **0.0954** | **0.0528** |
| SASRec — text (frozen) | 768→64 | 0.10M | 0.0874 | 0.0470 |

- **학습이 핵심이다**: 둘 다 정적 기준선을 ~3.5–4배 앞선다. 개인화는 사용자 시퀀스
  학습에서 나온다.
- 그러나 **이 조건에선 ID가 근소 우세**. **가설 미지지.** frozen text는 projection이
  전 아이템에 공유되어 희소 아이템을 개별 특화하지 못한다(단 16배 적은 파라미터로 ID의
  ~92%를 냄).

### 4. 다음 작업 / 요약
"frozen"이라는 제약이 원인일 수 있다 → 텍스트 임베딩을 학습 가능하게 풀어 재검증(실험 2).

---

# 실험 2 — 텍스트 미세조정 (`--train_text`)

### 1. 가설
frozen을 해제하고 텍스트 임베딩을 의미적 초기값으로 두고 end-to-end 학습하면, text가
ID를 앞설 것이다.

### 2. 실행 코드
```bash
uv run python3 train_sasrec.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --item_repr text --train_text --epochs 80 --patience 6 \
    --out results/sasrec_text_ft.json
```

### 3. 결과 및 결론
| 방법 | 차원 | 학습 파라미터 | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| SASRec — ID | 64 | 1.69M | 0.0954 | 0.0528 |
| **SASRec — text(ft)** | 768→64 | 19.7M | **0.1011** | **0.0563** |

- **text(ft)가 ID를 전 구간에서 앞선다** (전체 +6%, 수렴도 23 epoch로 더 빠름).
  표면적으로는 **가설 지지**.
- **그러나 용량 교란이 남는다**: text(ft)는 아이템당 768차원(19.7M), ID는 64차원(1.69M).
  승리가 "의미적 초기화" 때문인지 "더 큰 용량" 때문인지 분리되지 않았다.

### 4. 다음 작업 / 요약
용량을 맞춘 대조군이 필요 → 동일 구조·파라미터에 **무작위 초기화**한 768차원 테이블
(randproj)과 비교(실험 3).

---

# 실험 3 — 용량 대조군 (random-init 768)

### 1. 가설
text(ft)의 승리는 "더 큰 용량"이 아니라 "의미적 초기화" 덕분이다. → 동일 용량 무작위
초기화(randproj)보다 text(ft)가 나아야 한다.

### 2. 실행 코드
```bash
uv run python3 train_sasrec.py --data_dir data/video_games --item_repr randproj \
    --latent_dim 768 --epochs 80 --patience 6 --out results/sasrec_randproj.json
```

### 3. 결과 및 결론
| 방법 (모두 19.7M·768→64) | Recall@10 | NDCG@10 |
|---|---|---|
| randproj (무작위 768) | 0.0989 | 0.0552 |
| text(ft) (의미 768) | 0.1011 | 0.0563 |

인기도 구간별 Recall@10 (★ = randproj 대비 통계적 유의):
| 구간 | ID(64) | randproj(무작위) | text(ft)(의미) | 의미 init 효과 |
|---|---|---|---|---|
| cold [5,20) | 0.0251 | 0.0233 | **0.0287** | **+23% ★** |
| mid [20,100) | 0.0637 | 0.0626 | **0.0736** | **+18% ★** |
| hot [100,∞) | 0.1950 | **0.2078** | 0.1991 | −4% ★ (무작위 우세) |

- **전체 우위는 거의 전부 '용량' 효과다**: randproj(0.0989) ≈ text(ft)(0.1011), 차이
  0.0022는 ~1.6σ로 **무의미**. "텍스트 표현이 *전반적으로* 낫다"는 주장은 차원/용량
  이야기로 후퇴.
- **의미적 초기화의 진짜 효과는 데이터가 적은 곳에 집중**된다: cold(+23%)·mid(+18%)에서
  유의하게 우세. hot에선 데이터가 충분해 무작위도 충분(오히려 우세).
- **부분적 가설 지지**: 표현의 의미 정보는 *희소 아이템*에서만 유의하게 기여.

### 4. 다음 작업 / 요약
의미 우위가 데이터 희소 구간에 있다면, 그 극단인 **미학습(zero-shot) 아이템**에서
가장 분명히 드러나야 한다 → zero-shot 평가(실험 4).

---

# 실험 4 — Zero-shot (콜드 아이템 holdout)

### 1. 가설
텍스트 표현은 학습 중 한 번도 본 적 없는 아이템도 그 텍스트만으로 추천할 수 있다.
ID 표현은 (임베딩이 없어) 구조적으로 불가능하다.

### 2. 실행 코드
```bash
uv run python3 zeroshot.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --holdout_frac 0.10 --epochs 60 --patience 5 --out results/zeroshot.json
```
프로토콜: train 어휘의 10%를 무작위로 COLD로 지정해 **학습에서 완전히 제거**하고,
test 타깃이 COLD인 사용자에 대해 WARM ∪ COLD 전체를 후보로 평가. ID는 COLD 표현이
없고, text는 학습된 projection을 COLD 아이템의 frozen 텍스트 벡터에 적용해 랭킹.

### 3. 결과 및 결론
| 평가 | ID | text(frozen) |
|---|---|---|
| **cold 타깃 (zero-shot, n=10,167)** | **0.0** (구조적) | **0.0145** (≈12σ > 0) |
| warm 타깃 (참고, n=5,000) | 0.0828 | 0.0774 |

- **ID는 미학습 아이템에서 정확히 0**: 임베딩 행이 없어 추천 자체가 불가능.
- **text(frozen)는 텍스트만으로 Recall@10=0.0145** 달성 — 의미 공간이 미학습 아이템을
  올바른 사용자 근처에 실제로 배치. warm 성능(0.0774)의 ~19%를 zero-shot으로 회복.
- **가설 결정적 지지** (이 영역에서).

### 4. 다음 작업 / 요약
ID가 0인 곳에서 text가 유한한 추천 능력을 가진다는 점이, 텍스트 표현의 본질적 가치다.

---

## 전체 결론 — 가설 판정

가설 "텍스트 임베딩 표현이 더 나은 추천을 준다"는 **무조건 참도 거짓도 아니고, 조건부로
참**이다. 영역별로 갈린다:

| 영역 | 우세 표현 | 이유 |
|---|---|---|
| 정적(학습 없음) | — | 학습이 없어 표현 가치 측정 불가 |
| warm 아이템 (동일 용량) | ID ≈ text | 협업신호 암기로 충분 |
| 희소 cold/mid 아이템 | **text** (+18~23%) | 데이터 부족 → 의미 정보 기여 |
| 미학습 zero-shot 아이템 | **text** (0.0145 vs 0) | ID는 구조적으로 불가능 |

→ 실무적으로는 순수 ID도 순수 text도 아닌 **hybrid**(warm은 ID 암기, cold/zero-shot은
text 일반화)가 자연스러운 다음 방향이다.

## 다음 단계
- **hybrid** (`--item_repr hybrid`): 두 강점의 결합.
- **텍스트 인코더 end-to-end 미세조정**: per-item 벡터가 아니라 embeddinggemma 자체 학습.
- **cross-domain transfer**: 한 카테고리에서 학습한 text 표현을 다른 카테고리에 적용.

## 파일
- `preprocess.py` — 스트리밍 파싱, 반복적 k-core, leave-one-out
- `build_item_text.py` — item_id → 메타데이터 텍스트
- `embed_items.py` — embeddinggemma 아이템 인코딩 (GPU)
- `baseline_popularity.py` — 인기도 기준선 + 재사용 가능한 평가 하니스(`evaluate()`)
- `baseline_text_embed.py` — 공유 하니스 상의 텍스트 mean-pool score_fn
- `sasrec.py` — SASRec 백본 + 교체 가능한 아이템 표현(id / text / randproj / hybrid)
- `train_sasrec.py` — SASRec 학습/평가 (공유 하니스 채점, 인기도 구간별 분해)
- `zeroshot.py` — 콜드 아이템 holdout zero-shot 평가 (ID vs text 일반화)

## 결과 산출물 (`results/`)
`sasrec_id.json` · `sasrec_text.json` · `sasrec_text_ft.json` · `sasrec_randproj.json`
· `zeroshot.json` (+ 각 `.log`)
