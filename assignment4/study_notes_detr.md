# CS444 Assignment 4 — DETR 학습 정리노트

---

## 1. DETR이란 무엇인가 (핵심 개념)

### 1.1 기존 Object Detection의 문제

기존 방법(YOLO, Faster R-CNN)은 아래 hand-crafted 파이프라인이 필수였다:

- **Anchor box**: 수백~수천 개의 미리 정의된 박스 생성
- **NMS (Non-Maximum Suppression)**: 중복 예측 제거
- **복잡한 후처리**: 클래스별 임계값 조정 등

이런 요소들은 도메인 지식이 필요하고, end-to-end 학습이 어렵다.

### 1.2 DETR의 핵심 아이디어

> **Detection을 Set Prediction 문제로 재정의한다.**

- N개의 object를 한번에 예측 (병렬)
- anchor 없음, NMS 없음
- 완전한 end-to-end 학습 가능

---

## 2. DETR 아키텍처 전체 파이프라인 (★★★ 가장 중요)

```
입력 이미지들 (크기 다를 수 있음)
        ↓
pad_images_to_batch()
→ x:    [B, C, H_max, W_max]   (패딩된 배치)
→ mask: [B, H, W]              (True=패딩 픽셀, False=실제 픽셀)
        ↓
Backbone (DINOv2 ViT)
→ features: [B, out_channels, H', W']
             (H', W'는 patch 단위로 줄어든 해상도)
        ↓
input_proj (1×1 Conv)
→ src: [B, d_model, H', W']    (채널을 d_model=256으로 압축)
        ↓
mask interpolate (제공된 코드)
→ mask: [B, H', W']            (src 해상도에 맞게 다운샘플)
        ↓
PositionEmbeddingSine(src, mask)
→ pos: [B, d_model, H', W']   (src와 완전히 같은 shape)
        ↓
flatten + permute
→ src+pos: [B, H'*W', d_model]   ← Encoder 입력
→ mask:    [B, H'*W']             ← padding 마스크 (2D→1D)
        ↓                ↑ cross-attention
query_embed.weight              |
→ tgt: [B, num_queries, d_model] ← Decoder 입력
        ↓
nn.Transformer(src=src+pos, tgt=tgt, src_key_padding_mask=mask)
→ hs: [B, num_queries, d_model]
        ↓                    ↓
class_embed(hs)          bbox_embed(hs).sigmoid()
→ pred_logits:           → pred_boxes:
  [B, Q, num_classes+1]    [B, Q, 4] ∈ [0,1]
  (20 + 1개 클래스)         (cx, cy, w, h 정규화 좌표)
```

### 2.1 핵심 구현 포인트

| 포인트 | 내용 |
|---|---|
| `pos`는 `src`에 더함 | `src + pos` 후 Transformer 입력 |
| `mask`는 `src_key_padding_mask`로 전달 | Encoder가 패딩 위치 무시 |
| `bbox`에 sigmoid 필수 | 좌표를 [0,1]로 제한 |
| `batch_first=True` 필수 | PyTorch 기본값이 False (seq, B, C) |

---

## 3. 핵심 컴포넌트 상세 설명 (★★★)

### 3.1 Object Query (`query_embed`)

```python
self.query_embed = nn.Embedding(num_queries, d_model)
# num_queries=25: 이미지에서 찾을 최대 객체 수
# d_model=256:    각 query의 벡터 크기
```

**가장 많이 헷갈리는 개념:**

- 이미지와 **무관한** 학습 가능한 파라미터
- CNN/DINOv2로 인코딩한 결과가 **아님**
- DETR이 추가한 고유 설계 — PyTorch `nn.Transformer` 내부 컴포넌트가 아님
- Decoder의 `tgt` 자리에 들어가서, **cross-attention을 통해** 이미지 정보를 흡수함

**비유:** N명의 탐정(query)이 수사 파일(이미지 feature)을 읽으며 각자 담당 물체를 찾는 것

```python
# forward에서
tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
# weight: [Q, d_model] → unsqueeze → [1, Q, d_model] → expand → [B, Q, d_model]
# expand: 메모리 복사 없이 B차원 반복처럼 보이게 함 (repeat보다 효율적)
```

### 3.2 Positional Encoding

Transformer는 순서 개념이 없다. 이미지를 flatten하면 위치 정보가 사라지므로,
2D 위치를 sine/cosine 함수로 인코딩해서 feature에 더해준다.

```python
# PositionEmbeddingSine: x, y 좌표를 각각 sin/cos 인코딩
# 출력: [B, d_model, H', W'] — src와 완전히 같은 shape
pos = self.position_embedding(src, mask)
# flatten + permute 후 src에 더함
```

### 3.3 Padding Mask

배치 내 이미지 크기가 달라서 가장 큰 이미지 기준으로 0-패딩을 채운다.
패딩된 부분은 가짜 정보이므로 Transformer attention에서 무시해야 한다.

```
mask[b, i, j] = True  → 패딩 픽셀, attention 무시
mask[b, i, j] = False → 실제 픽셀, attention 허용
```

```python
# flatten 후 src_key_padding_mask로 전달
mask = mask.flatten(1)   # [B, H', W'] → [B, H'*W']
self.transformer(src=..., src_key_padding_mask=mask)
```

### 3.4 주요 레이어 비교

| 레이어 | 용도 | 학습 가능 여부 |
|---|---|---|
| `nn.Embedding(N, D)` | 정수 인덱스 → 벡터 (lookup table) | ✓ |
| `nn.Linear(in, out)` | 벡터 → 벡터 (행렬곱 y=xW+b) | ✓ |
| `nn.Conv2d(in, out, 1)` | feature map 채널 변환 | ✓ |
| `MLP` | Linear + ReLU 여러 층 (= Fully Connected Network) | ✓ |

**MLP (Multi-Layer Perceptron):**
- Perceptron = Linear + activation 1개
- MLP = Perceptron 여러 개를 직렬로 쌓은 것
- bbox_embed는 비선형 회귀가 필요해서 MLP 사용, class_embed는 Linear 1개로 충분

---

## 4. Hungarian Matching Loss (★★★ 구현 핵심)

### 4.1 왜 필요한가

N개 예측 ↔ M개 GT (M ≤ N), 어떻게 매칭?
단순 순서 매칭 불가 → 최적 이분 매칭 필요

### 4.2 Hungarian Algorithm

비용 행렬 C [Q × T] 를 최소화하는 순열 σ 탐색:

```
cost_class = -pred_prob[:, tgt_ids]        # 높은 확률 → 낮은 비용
cost_bbox  = L1(pred_boxes, tgt_boxes)     # pairwise L1 [Q, T]
cost_giou  = -GIoU(pred_boxes, tgt_boxes)  # 높은 GIoU → 낮은 비용

C = λ_bbox * cost_bbox + λ_cls * cost_class + λ_giou * cost_giou
src_idx, tgt_idx = linear_sum_assignment(C)  # scipy
```

결과: 각 이미지에 대해 `(src_idx, tgt_idx)` 쌍 → "query i번 ↔ GT j번"

### 4.3 3가지 Loss 함수

**① Classification Loss (loss_ce)**

```python
# 전체 query를 no-object(=num_classes=20)로 초기화
target_classes = torch.full([B, Q], num_classes)
# matched 위치만 실제 GT label로 덮어쓰기
target_classes[batch_idx, src_idx] = tgt_classes_matched
# CE loss with class weight (no-object class는 0.1 또는 0.2 가중치)
loss_ce = F.cross_entropy(pred_logits.transpose(1,2), target_classes, empty_weight)
```

`eos_coef (empty_weight[-1])`: no-object class의 가중치. 낮을수록 배경을 덜 학습.
이미지 대부분의 query가 no-object이므로 가중치를 낮추지 않으면 배경만 잘 맞추는 모델이 됨.

**② L1 Box Loss (loss_bbox)**

```python
# matched pair만 대상
src_boxes = pred_boxes[batch_idx, src_idx]   # [N_matched, 4]
tgt_boxes = cat([t["boxes"][j] for ...])     # [N_matched, 4]
loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
```

**③ GIoU Loss (loss_giou)**

```
GIoU(A, B) = IoU(A, B) - (|C| - |A∪B|) / |C|
```
- C: A와 B를 포함하는 최소 enclosing box
- IoU가 0일 때도 gradient 제공 (IoU loss의 단점 보완)
- 범위: (-1, 1]

```python
# cxcywh → xyxy 변환 필수
giou_matrix = generalized_box_iou(
    box_cxcywh_to_xyxy(src_boxes),
    box_cxcywh_to_xyxy(tgt_boxes)
)
# 대각선만 사용 (이미 matched pair이므로 [i,i]가 pair)
loss_giou = (1 - giou_matrix.diag()).sum() / num_boxes
```

**최종 loss:**
```python
weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
total_loss = compute_total_loss(loss_dict, weight_dict)
```

---

## 5. 구현 시 버그 포인트 (★★)

| 버그 | 원인 | 수정 |
|---|---|---|
| `backbone.num_channels` | 존재하지 않는 속성 | `backbone.out_channels` |
| `nn.Transformer()` 기본값 | batch_first=False → shape 불일치 | `batch_first=True` 추가 |
| `src`만 Transformer 입력 | positional encoding 누락 | `src + pos` 로 변경 |
| bbox sigmoid 누락 | 좌표가 [0,1] 범위 벗어남 | `.sigmoid()` 추가 |
| `num_boxes=0` | 빈 배치 시 나눗셈 에러 | `max(num_boxes, 1)` |

---

## 6. 훈련 설정 (★★)

### 6.1 YOLO와의 차이

| | YOLO | DETR |
|---|---|---|
| Optimizer | SGD | AdamW |
| LR | 상대적으로 높음 | 낮음 (5e-5 ~ 1.5e-4) |
| Backbone LR | 동일 | Head보다 낮게 (3:1) |
| 학습 속도 | 빠름 | 느림 |

### 6.2 훈련 트릭

```python
# 1. Backbone freeze: 처음 2 epoch 동안 backbone lr=0
freeze_backbone_epochs = 2.0

# 2. Warmup: 첫 0.5 epoch 동안 lr 선형 증가
warmup_epochs = 0.5

# 3. Cosine scheduler: lr을 점진적으로 감소
CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)

# 4. Gradient clipping: 폭발적 gradient 방지
clip_grad_norm_(model.parameters(), 0.1)
```

### 6.3 소요 시간 예상 (15 epochs, VOC 2007 ~5011장)

| 환경 | 예상 시간 |
|---|---|
| Colab T4 | 1.5 ~ 2.5시간 |
| Colab A100 | 40분 ~ 1시간 |
| CPU | 사실상 불가 |
| TPU (v5e) | 비추천 (scipy Hungarian이 CPU 전용) |

---

## 7. 성능 목표 달성 전략 (★)

목표: **mAP ≥ 0.58** (15 epochs)

**1순위 — 구현 정확성 (가장 중요)**
교수님이 하이퍼파라미터를 이미 0.58 달성 기준으로 세팅함.
구현이 맞으면 기본값으로도 통과 가능.

**2순위 — GPU 사용 필수**

**3순위 — 그래도 안 되면 튜닝할 것들**

| 파라미터 | 기본값 | 시도값 |
|---|---|---|
| `num_queries` | 12 | 15~25 |
| `eos_coef` | 0.2 | 0.1 |
| `num_decoder_layers` | 3 | 4 |
| `freeze_backbone_epochs` | 2.0 | 1.0 |

---

## 8. DETR vs 기존 방법 비교표 (★)

| | YOLO/Faster R-CNN | DETR |
|---|---|---|
| Anchor | 필요 | 없음 |
| NMS | 필요 | 없음 |
| 인코딩 | CNN only | CNN/ViT + Transformer |
| Loss | per-anchor | Hungarian set matching |
| 장거리 관계 | 어려움 | Self-attention으로 자연스럽게 |
| 학습 속도 | 빠름 | 느림 |
| 구현 복잡도 | 높음 (anchor, NMS 설계) | 낮음 (end-to-end) |

---

## 9. 제출 체크리스트

```
□ detr.py 구현 완료 및 버그 수정
□ detr_loss.py 구현 완료
□ Colab GPU로 15 epoch 훈련
□ mAP >= 0.58 확인
□ my_solution_detr.csv 생성
□ Kaggle DETR competition 제출 (Submit 버튼 클릭)
□ mp4b_detr_output.pdf (노트북 출력 포함 PDF)
□ netid_mp4_code.zip (detr.py, detr_loss.py 포함)
□ netid_mp4_report.pdf
```
