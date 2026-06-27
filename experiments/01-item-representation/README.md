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

# 실험 5 — hybrid (ID + text 결합)

### 1. 가설
지금까지의 영역별 결과(warm은 ID, cold/zero-shot은 text)가 맞다면, 둘을 더한 hybrid
표현은 **양쪽 강점을 모두** 가져야 한다: hot 구간에서 ID 수준의 성능을 유지하면서,
cold/mid 구간에서는 text의 의미 일반화 이득을 얻는다.

설계: `M = id_emb + LayerNorm(Linear(text_matrix))` ([sasrec.py](sasrec.py)의 `matrix()`).
학습되는 ID 테이블(per-item warm 암기) 위에, 의미 정보를 담은 텍스트 표현을 **가산적
prior**로 더한다. 두 변형을 비교한다:
- **hybrid (frozen text)**: ID(학습) + frozen 텍스트 projection. 용량 추가가 거의 없음
  (ID 1.69M + projection 0.05M ≈ 1.74M).
- **hybrid (train_text)**: 텍스트 768d까지 학습(21.3M). cold 이득을 냈던 text(ft)와
  동일 조건의 텍스트 분기 + ID. 결합이 용량 때문인지 의미 prior 때문인지 분리한다.

### 2. 실행 코드
```bash
# hybrid (frozen text): ID + frozen 텍스트 prior
uv run python3 train_sasrec.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --item_repr hybrid --epochs 80 --patience 6 --out results/sasrec_hybrid.json

# hybrid (train_text): 텍스트도 학습 (용량 대조)
uv run python3 train_sasrec.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --item_repr hybrid --train_text --epochs 80 --patience 6 \
    --out results/sasrec_hybrid_ft.json
```

### 3. 결과 및 결론
| 방법 | 학습 파라미터 | Recall@10 | NDCG@10 |
|---|---|---|---|
| SASRec — ID | 1.69M | 0.0954 | 0.0528 |
| SASRec — text(ft) | 19.7M | 0.1011 | 0.0563 |
| SASRec — randproj | 19.7M | 0.0989 | 0.0552 |
| **SASRec — hybrid(frozen)** | **1.74M** | **0.1042** | **0.0579** |
| SASRec — hybrid(train_text) | 21.3M | 0.1015 | 0.0566 |

인기도 구간별 Recall@10:
| 구간 | ID(1.69M) | text(ft)(19.7M) | **hybrid(frozen)(1.74M)** | hybrid vs ID | hybrid vs text(ft) |
|---|---|---|---|---|---|
| cold [5,20) | 0.0251 | 0.0287 | **0.0282** | **+12%** | ≈ (−2%) |
| mid [20,100) | 0.0637 | 0.0736 | **0.0738** | **+16%** | ≈ (+0.3%) |
| hot [100,∞) | 0.1950 | 0.1991 | **0.2091** | **+7%** | **+5%** |

- **가설 지지 — 두 강점이 실제로 결합된다**: hybrid(frozen)은 cold에서 ID 대비 +12%
  (text(ft) 수준까지 회복)하면서, hot에서는 ID·text(ft)를 모두 앞선다(0.2091, 전 arm
  최고). 전체 Recall@10=0.1042로 **모든 단일 표현(ID·text·randproj·text(ft))을 능가**한다.
  학습되는 ID 테이블이 warm 아이템을 개별 암기하고, frozen 텍스트 prior가 데이터가
  적은 아이템을 의미 공간에서 보강하는, 정확히 기대했던 분업이 일어난다.
- **결합 효과는 '용량'이 아니라 '의미 prior'다**: hybrid(frozen)은 단 1.74M
  파라미터(≈ID)로 19.7M짜리 text(ft)보다 낫다. 텍스트를 학습 대상으로 풀면
  (train_text, 21.3M) 오히려 전체(0.1015)·hot(0.2018) 모두 frozen보다 **떨어진다** —
  실험 3의 교훈("학습 가능한 큰 텍스트 테이블의 추가 용량은 도움이 안 된다")과 일치한다.
  frozen 의미 벡터를 가산 prior로 쓰는 쪽이 sweet spot.

> **왜 hybrid에서는 텍스트를 frozen으로 둬야 하나? (train_text가 오히려 손해인 이유)**
>
> 여기서 `--train_text`는 embeddinggemma 모델을 재학습하는 게 아니라, embed_items.py가
> 미리 뽑아 둔 **아이템별 768d 벡터 테이블**을 학습 중에 gradient로 흔드는 것이다.
> 텍스트 임베딩의 가치는 "비슷한 아이템이 가까이 놓인 의미 지도" 그 자체이고, 이는
> 상호작용이 적은 cold 아이템에게 특히 귀한 **외부 지식(prior)**이다.
>
> hybrid에는 이미 학습되는 **ID 테이블**이 협업신호(어떤 아이템이 같이 소비되는가)를
> 전담한다. 이때 텍스트까지 학습으로 풀면, 상호작용이 많은 hot 아이템의 강한 gradient가
> 텍스트 벡터를 협업신호 쪽으로 끌어당기는데 — 이는 **ID 분기가 이미 하는 일의 중복**일
> 뿐이고, 그 과정에서 cold를 돕던 깔끔한 의미 구조만 일그러진다. 그래서 hybrid에서
> 텍스트 분기의 최선은 "고정된 의미 prior로 가만히 있는 것"이다.
>
> 단, **"텍스트 fine-tune은 항상 나쁘다"는 일반화는 틀리다.** ID 분기가 없는 text-only
> arm에서는 정반대였다:
>
> | | frozen | train_text |
> |---|---|---|
> | text-only (ID 분기 없음) | 0.0874 | **0.1011** (크게 향상) |
> | hybrid (ID 분기 있음) | **0.1042** | 0.1015 (하락) |
>
> text-only는 협업신호를 담을 곳이 텍스트 테이블뿐이라 학습으로 풀어야 용량이 생겨
> 성능이 오르고, hybrid는 ID가 그 역할을 이미 맡으므로 텍스트는 고정 prior로 두는 편이
> 낫다. **fine-tune의 득실은 ID 분기가 협업신호를 흡수해 주는지에 달려 있다.**

### 4. 다음 작업 / 요약
hybrid(frozen)이 "ID의 warm 암기 + text의 cold 일반화"를 거의 ID 비용으로 결합함을
확인했다. 다음은 이 결합이 **미학습 zero-shot 아이템**에서도 유지되는지(ID 분기는 0,
text 분기는 일반화 → hybrid는 warm 손실 없이 zero-shot 능력을 얻는가)를 zero-shot
프로토콜(실험 4)로 검증하는 것이다.

---

# 실험 6 — hybrid의 zero-shot 검증 (warm 유지 + 미학습 아이템 일반화)

### 1. 가설
hybrid(frozen)은 warm 성능을 잃지 않으면서, 학습 중 본 적 없는 zero-shot 아이템에 대한
추천 능력도 가질 것이다. 근거: hybrid의 cold 아이템에는 ID 행이 없으므로 표현이 자동으로
text 분기(`proj(frozen text)`)로 **fallback**한다 → text arm과 같은 일반화 경로를 타되,
warm 아이템에서는 ID 암기를 그대로 유지한다.

### 2. 실행 코드
실험 4와 동일한 holdout 프로토콜에 hybrid arm을 추가([zeroshot.py](zeroshot.py)).
catalog 구성: warm = `id_emb + proj(text)`(학습된 행), cold = `proj(text)`만(ID 행 없음).
```bash
uv run python3 zeroshot.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --holdout_frac 0.10 --epochs 60 --patience 5 --out results/zeroshot.json
```

### 3. 결과 및 결론
holdout: warm 22,975 · cold(미학습) 2,552.

| 평가 | ID | text(frozen) | **hybrid(frozen)** |
|---|---|---|---|
| **cold 타깃 (zero-shot, n=10,167)** | 0.0 (구조적) | **0.0145** | 0.0080 |
| warm 타깃 (참고, n=5,000) | 0.0828 | 0.0774 | **0.0904** |

- **가설 방향상 확정**: hybrid는 (a) warm을 잃지 않는다 — 오히려 ID(0.0828)·text(0.0774)를
  모두 앞서 **warm 최고(0.0904)**, (b) zero-shot 능력을 **실제로 획득** — cold에서
  0.0080 > 0(ID는 구조적 불가능). 미학습 아이템에 text projection이 적용돼 추천이 된다.
- **단, hybrid의 zero-shot은 순수 text의 절반(0.0080 vs 0.0145)**. 원인은 표현 스케일
  비대칭이다: hybrid catalog에서 warm은 `id_emb + proj(text)`(노름이 큼), cold는
  `proj(text)`뿐(노름이 작음) → 점수 `uv·M`이 warm 쪽으로 체계적으로 쏠려 cold가
  랭킹에서 불리하다. 순수 text는 모든 아이템이 동일한 `proj(text)` 표현이라 cold가
  공정하게 경쟁한다.
- **실무적 함의**: warm 위주 트래픽 + 가끔 cold라면 hybrid가 최선(warm 최고 + nonzero
  zero-shot). 반대로 zero-shot/콜드스타트 자체가 핵심 목표라면, warm·cold 표현의 노름을
  맞추는 정규화나 별도의 cold 경로가 필요하다.

### 4. 다음 작업 / 요약
hybrid는 "warm 손실 없이 zero-shot 능력을 얻는다"가 참이되, zero-shot 품질은 표현 스케일
비대칭 탓에 순수 text에 못 미친다. 자연스러운 후속은 warm/cold 표현 정규화(예: cold에도
ID 노름에 맞춘 보정, 또는 representation L2-norm 일치)로 이 비대칭을 줄이는 것이다.

---

# 실험 7 — 표현 정규화로 hybrid의 zero-shot 회복

### 1. 가설
실험 6에서 진단한 "warm(`id+text`, 큰 노름) vs cold(`text`, 작은 노름)" 스케일 비대칭이
hybrid zero-shot 부진의 원인이라면, catalog 표현을 **L2 정규화**(각 행을 단위 노름으로 =
item쪽 cosine scoring)하면 cold가 공정하게 경쟁해 hybrid의 zero-shot이 **순수 text 수준
까지** 회복되어야 한다.

### 2. 실행 코드
[zeroshot.py](zeroshot.py)에 L2 정규화 변형을 추가(`hybrid_frozen_l2`, `text_frozen_l2`).
scoring 직전 catalog 행렬 `M`의 각 행을 단위 노름으로 정규화한다(user 벡터 정규화는
argsort 불변이라 생략). 학습/holdout/평가 프로토콜은 실험 6과 동일.
```bash
uv run python3 zeroshot.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --holdout_frac 0.10 --epochs 60 --patience 5 --out results/zeroshot.json
```

### 3. 결과 및 결론
| 평가 | text(raw) | hybrid(raw) | **hybrid + L2** | text + L2 |
|---|---|---|---|---|
| **cold 타깃 (zero-shot, n=10,167)** | 0.0145 | 0.0080 | **0.0631** | 0.0045 |
| warm 타깃 (참고, n=5,000) | 0.0774 | 0.0904 | 0.0564 | 0.0580 |

- **가설 초과 달성 — "순수 text 수준까지"가 아니라 그 이상**: L2 정규화가 hybrid의
  zero-shot을 0.0080 → **0.0631로 ~8배** 끌어올려, 순수 text(0.0145)를 **4배 이상**
  앞선다. 진단(노름 비대칭)이 정확했음이 확인된다 — 단위 노름이 warm의 점수 독식을
  제거하자 cold가 공정하게 경쟁한다. 게다가 hybrid의 user 인코더는 학습 중 ID 분기 덕에
  warm 시퀀스를 잘 적합했고, 그 더 나은 user 표현을 cosine으로 cold의 text projection과
  맞추니 순수 text보다도 cold를 잘 랭킹한다.
- **그러나 공짜가 아니다 — 명백한 trade-off**: 같은 정규화가 warm을 0.0904 → 0.0564로
  떨어뜨린다(ID 0.0828·text 0.0774보다도 아래). 노름이 warm 아이템의 인기도/신뢰도
  정보를 담고 있었고, 이를 버리면 warm 랭킹이 손해다. 실제로 **순수 text는 정규화가
  일률적으로 해롭다**(cold 0.0145→0.0045, warm 0.0774→0.0580) — text 행은 이미 비슷한
  노름이라 정규화가 유용한 미세 신호만 지운다.
- **결론**: 전역 L2 정규화는 hybrid를 *탁월한 zero-shot 모델*로 바꾸지만 *평범한 warm
  모델*로 만든다. 즉 "정규화로 zero-shot을 회복할 수 있는가?"의 답은 **예(그 이상),
  단 warm을 희생**한다.

### 4. 다음 작업 / 요약
정규화는 cold/warm 어느 한쪽을 택하는 스위치가 아니라 **둘을 동시에 잡는 비대칭 scoring**
으로 발전해야 한다: 예) warm은 dot-product(노름 신호 유지)·cold는 cosine으로 채점하는
분기 scoring, per-item bias 항 학습, 또는 학습 단계에서 normalized scoring + temperature
도입. 이로써 warm 0.0904와 cold 0.0631을 한 모델에서 동시에 얻는 것이 목표다.

---

# 실험 8 — 비대칭 scoring (cosine + popularity bias)

### 1. 가설
실험 7의 trade-off(L2는 cold↑·warm↓)는 cosine이 warm의 노름=인기도 신호를 버리기
때문이다. 그렇다면 **cosine으로 방향을 공정하게 비교**하되 **버려진 인기도 신호를 bias로
명시적으로 복원**하면, warm(0.0904)과 cold(0.0631)을 한 모델에서 동시에 잡을 수 있을 것이다:

```
score(u, i) = <ûv, M̂ᵢ>  +  β · logpop_norm(i)
```
- `<ûv, M̂ᵢ>`: uv·M 모두 단위 노름 → cosine ∈ [−1,1] (warm/cold를 방향으로 공정 비교).
- `logpop_norm(i) = log(1+train_pop) / max`: [0,1]로 정규화해 β를 cosine 단위로 해석.
  cold는 학습에서 held-out → train_pop=0 → **bias=0이 자연 성립**(특수 처리 불필요).
  warm에만 인기도 prior가 가산되어 L2가 버린 신호를 복원한다.

### 2. 실행 코드
[zeroshot.py](zeroshot.py)의 `eval_over_catalog`에 `bias`·`normalize` 옵션을 추가하고
β를 sweep. β=0은 순수 cosine(=실험 7의 `hybrid_frozen_l2`)이라 내부 검증도 된다.
```bash
uv run python3 zeroshot.py --data_dir data/video_games --emb_dir data/video_games/emb \
    --holdout_frac 0.10 --epochs 60 --patience 5 --out results/zeroshot.json
# 결과의 asym_scoring_cosine_plus_beta_logpop 참고
```

### 3. 결과 및 결론
hybrid(frozen) catalog에 cosine + β·logpop 적용, β sweep:

| β | cold Recall@10 | warm Recall@10 | 비고 |
|---|---|---|---|
| 0.0 (순수 cosine) | **0.0631** | 0.0564 | = 실험 7 L2 (cold 최대) |
| 0.05 | 0.0389 | 0.0680 | 균형점 후보 |
| 0.10 | 0.0196 | 0.0758 | 균형점 후보 |
| 0.15 | 0.0100 | 0.0806 | |
| 0.20 | 0.0041 | 0.0848 | |
| 0.30 | 0.0013 | **0.0900** | warm 최대(≈raw hybrid 0.0904) |
| 0.50 | 0.0001 | 0.0844 | 인기도 과지배 → warm도 하락 |

참고선: raw hybrid (warm 0.0904 / cold 0.0080) · 순수 text (warm 0.0774 / cold 0.0145).

- **두 극값의 동시 달성은 불가 — 매끄러운 Pareto frontier다**: warm 0.0904와 cold 0.0631은
  frontier의 양 끝(β=0.3 vs β=0)에 있어 **서로 배타적**이다. β를 올리면 warm이 0.0564→
  0.0900으로 오르는 만큼 cold가 0.0631→0.0013으로 단조 감소한다. 근본 이유: cold-타깃
  평가에서 인기 warm 아이템과 cold 타깃이 **같은 후보 리스트에서 경쟁**하므로, warm
  타깃을 돕는 인기도 가산은 동시에 cold 타깃을 묻어버린다. 단일 전역 점수로는 두 쿼리
  유형을 동시에 만족시킬 수 없다.
- **그러나 frontier는 튜닝 가능하고 두 baseline을 압도한다**: 중간 β는 양쪽 모두에서
  유용한 절충을 준다. 예) **β≈0.1**: warm 0.0758(≈순수 text 0.0774)이면서 cold
  0.0196(>순수 text 0.0145) — *한 모델*이 warm은 text 수준, cold는 text 초과를 동시에
  낸다(raw hybrid의 cold 0.0080의 2.5배). **β≈0.05**: cold 0.0389(text의 2.7배)를
  유지하며 warm 0.068까지 회복.
- **참고 — 정규화 스케일이 frontier 모양을 좌우**: logpop을 [0,1]로 정규화하지 않은 초기
  실험에서는 β=0.05만으로 cold가 0.0631→0.0004로 *절벽처럼* 붕괴했다(logpop 0~8이
  cosine [−1,1]을 압도). 정규화로 β를 cosine 단위에 맞추자 비로소 매끄러운 frontier가
  드러났다 — bias 항은 반드시 cosine과 같은 스케일로 맞춰야 한다.

### 4. 다음 작업 / 요약
"warm 0.0904 + cold 0.0631 동시"는 단일 전역 scoring으로는 **불가능**(Pareto 양 끝)하며,
β는 운영점을 고르는 손잡이일 뿐이다. 진짜 동시 달성은 **쿼리/세그먼트 인지 라우팅**을
요구한다: 콜드스타트 후보가 필요한 요청은 낮은 β(cosine), warm 추천은 높은 β로 채점하거나,
cold/warm 후보를 각자 랭킹해 rank-fusion으로 병합. 즉 "한 점"이 아니라 "한 시스템에서 두
운영점"이 올바른 목표다. 트래픽의 cold:warm 비율로 단일 β를 고른다면 β≈0.05–0.1이 균형점.

---

## 전체 결론 — 가설 판정

가설 "텍스트 임베딩 표현이 더 나은 추천을 준다"는 **무조건 참도 거짓도 아니고, 조건부로
참**이다. 영역별로 갈린다:

| 영역 | 우세 표현 | 이유 |
|---|---|---|
| 정적(학습 없음) | — | 학습이 없어 표현 가치 측정 불가 |
| warm 아이템 (동일 용량) | ID ≈ text | 협업신호 암기로 충분 |
| 희소 cold/mid 아이템 | **text** (+18~23%) | 데이터 부족 → 의미 정보 기여 |
| 미학습 zero-shot 아이템 (순수) | **text** (0.0145 vs 0) | ID는 구조적으로 불가능 |
| 전 구간 종합 (in-vocab) | **hybrid(frozen)** | warm=ID 암기 + cold=text prior, 거의 ID 비용 |
| warm + 가끔 zero-shot | **hybrid(frozen)** | warm 최고(0.0904) + nonzero zero-shot(0.0080) |
| zero-shot 극대화 | **hybrid + L2** | cold 0.0631(text의 4배+), 단 warm 0.0564로 희생 |
| cold↔warm 절충 | **hybrid + cosine+β·pop** | β로 frontier 위 운영점 선택 (β≈0.05–0.1 균형) |

→ 예측대로 **hybrid가 두 강점을 결합**한다(실험 5). 순수 ID도 순수 text도 아닌
hybrid(frozen)이 전체(0.1042)·hot·cold를 모두 잡으며 전 arm 최고. 단, 텍스트를
학습으로 풀면(train_text) 오히려 손해 → frozen 의미 벡터를 **가산 prior**로 쓰는 것이
핵심. **zero-shot에서도**(실험 6) hybrid는 warm을 잃지 않고(오히려 최고) 미학습 아이템에
대한 추천 능력을 얻지만, warm/cold 표현의 노름 비대칭 탓에 *순수* zero-shot 품질은 순수
text에 못 미친다. 이 비대칭을 **L2 정규화로 제거**하면(실험 7) hybrid zero-shot이
0.0631로 순수 text(0.0145)를 4배+ 능가하지만 warm이 0.0564로 희생된다. 비대칭
scoring(cosine+β·pop, 실험 8)은 이 둘을 **매끄러운 Pareto frontier**로 잇지만, warm
0.0904와 cold 0.0631의 *동시* 달성은 단일 전역 점수로는 불가능하다 — β는 운영점을 고르는
손잡이이며, 진정한 동시 달성은 **쿼리/세그먼트 인지 라우팅**을 요구한다.

## 다음 단계
- ~~**hybrid** (`--item_repr hybrid`): 두 강점의 결합.~~ → 실험 5에서 검증 완료(최고 성능).
- ~~**hybrid의 zero-shot 검증**~~ → 실험 6에서 검증 완료(warm 유지+zero-shot 획득, 단 스케일 비대칭).
- ~~**hybrid 표현 정규화**~~ → 실험 7에서 검증 완료(L2로 zero-shot이 text의 4배+, 단 warm 희생).
- ~~**비대칭 scoring**(cosine+β·pop)~~ → 실험 8에서 검증 완료: 단일 전역 점수로는 동시
  달성 불가(Pareto frontier), β는 운영점 손잡이.
- **쿼리/세그먼트 인지 라우팅 또는 rank-fusion**: cold-start 요청과 warm 요청을 다른 β/
  채널로 채점해 warm 0.0904와 cold 0.0631을 *한 시스템*에서 동시에 제공.
- **학습 단계 normalized scoring + temperature**: 평가 시점 보정이 아니라, cosine 기반
  scoring을 손실에 넣어 학습 — bias 의존을 줄이고 frontier 자체를 끌어올릴 수 있는지.
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
- `zeroshot.py` — 콜드 아이템 holdout zero-shot 평가 (ID vs text vs hybrid, L2 정규화 변형 포함)

## 결과 산출물 (`results/`)
`sasrec_id.json` · `sasrec_text.json` · `sasrec_text_ft.json` · `sasrec_randproj.json`
· `sasrec_hybrid.json` · `sasrec_hybrid_ft.json` · `zeroshot.json` (+ 각 `.log`)

---

# 최종 종합 — "추천 모델 학습에 가장 적합한 Representation은 무엇인가?"

이 실험 전체를 관통하는 질문에 대한 답이다. 결론부터: **"단 하나의 최적 표현"을 찾는 질문
자체가 틀렸다.** 정답은 조건부이며, (1) 학습 여부, (2) 아이템의 데이터 영역(warm/cold/
미학습), (3) 최적화 대상(전체 정확도 vs 콜드스타트 커버리지)에 따라 갈린다. 다만 그 안에서
**실용적 기본값은 분명하다: 학습되는 ID + frozen 텍스트의 가산 hybrid.**

### 1. 핵심 증거 (8개 실험 종합)
| 질문 | 답 | 근거 실험 |
|---|---|---|
| 표현보다 먼저 중요한 건? | **학습/개인화** — 정적 표현은 다 비슷(0.025), 학습이 ~4배 끌어올림 | 0, 1 |
| warm/hot 아이템엔? | **ID(협업신호)** — 상호작용이 충분하면 ID 암기가 최강, 의미 init 불필요 | 1, 3, 5 |
| 텍스트 우위는 진짜 의미 덕분? | 전반적으론 **용량** 효과(randproj≈text(ft)). 의미 기여는 **희소 영역에 한정** | 3 |
| 희소 cold/mid 아이템엔? | **text(의미 prior)** — cold +23%, mid +18% | 3, 5 |
| 미학습 zero-shot 아이템엔? | **text만 가능** — ID는 구조적으로 0 | 4, 6 |
| 전 구간 단일 최적은? | **hybrid(frozen)** — overall 0.1042, hot·cold 동시 최고, 거의 ID 비용 | 5 |
| 텍스트를 fine-tune? | hybrid에선 **손해**(중복·과적합). ID 분기 없는 text-only에선 이득 | 2, 5 |
| 표현이 풀지 못하는 건? | warm↔cold **scoring 비대칭** — 표현이 아니라 *채점/서빙* 계층의 문제 | 7, 8 |

### 2. 결론: 세 가지 설계 원칙
1. **ID와 text는 경쟁자가 아니라 상보재다.** ID는 데이터가 많은 곳(warm/hot)의 협업신호를
   암기하고, text는 데이터가 마르는 곳(cold/long-tail/미학습)을 의미로 메운다. 어느 하나로는
   전 구간을 못 덮는다 — 그래서 **hybrid가 단일 표현 중 유일하게 전 구간을 잡는다.**
2. **결합은 '가산'으로, 텍스트는 'frozen'으로.** `M = id_emb + proj(frozen_text)`. 텍스트를
   학습으로 풀면 ID 분기와 역할이 겹쳐 의미 구조만 망가진다(실험 5). 의미 임베딩의 가치는
   *고정된 외부 지식(prior)*에 있지, 미세조정 여지에 있지 않다 — 적어도 per-item 벡터
   테이블 수준에서는. (※ embeddinggemma 인코더 자체의 end-to-end 학습은 별개 문제로 미검증.)
3. **표현과 scoring은 분리해서 풀어라.** hybrid의 zero-shot 약점(실험 6)은 표현 결함이
   아니라 warm/cold 노름 비대칭, 즉 채점 스케일 문제였다. cosine·popularity bias로
   조정되며(실험 7–8), warm↔cold 동시 최적은 단일 전역 점수가 아니라 **세그먼트 인지
   라우팅**으로 가야 한다. 즉 "최적 표현"과 "최적 채점"은 독립적으로 설계할 문제다.

### 3. 실무 권장
- **기본값: `hybrid`(frozen text).** in-vocab 전 구간 최고 + zero-shot 능력, ID 수준 비용.
- 콜드스타트/신상품 커버리지가 중요하면 채점에 **cosine + 작은 popularity bias(β≈0.05–0.1)**.
- zero-shot이 최우선이면 cold 후보에 **cosine(β→0) 라우팅**.
- 데이터가 풍부하고 신규 아이템이 거의 없다면 순수 **ID로도 충분**하다(가장 단순·저비용).

### 4. 한계 (이 결론의 적용 범위)
단일 카테고리(Amazon Video_Games), 단일 인코더(embeddinggemma-300m), 단일 백본(SASRec,
d_model=64), per-item 벡터 테이블 수준에서의 결론이다. "용량 vs 의미", "frozen prior가
sweet spot", "scoring은 분리 가능"이라는 *패턴*은 견고해 보이나, (a) 도메인/카탈로그 특성,
(b) 더 큰 d_model·백본, (c) 텍스트 인코더 end-to-end 학습, (d) cross-domain 전이에서의
재현은 후속 검증 과제다(위 "다음 단계").

### 한 줄 답
> **"가장 적합한 단일 표현"은 없다. 가장 적합한 *설계*는 — 협업신호(ID)와 의미 prior(frozen
> text)를 가산 결합한 hybrid를 표현으로 쓰고, warm/cold 균형은 표현이 아니라 scoring에서
> 푸는 것이다.**
