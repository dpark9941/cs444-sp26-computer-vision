# CS444 Assignment 4 — YOLO 학습 정리노트

---

## 1. YOLO란 무엇인가 (핵심 개념)

### 1.1 핵심 아이디어

> **이미지를 그리드로 나눠서, 각 셀이 직접 bbox와 클래스를 예측한다.**

- "You Only Look Once" — 이미지를 한 번만 보고 예측
- 단일 CNN forward pass로 모든 박스 동시 예측 (real-time 가능)
- 이 과제는 YOLO v1 기반 simplified 버전

### 1.2 DETR과의 근본적 차이

| | YOLO | DETR |
|---|---|---|
| 예측 방식 | Grid cell 기반 anchor | Object query 기반 |
| 후처리 | NMS 필요 | NMS 불필요 |
| Loss | per-cell 직접 매칭 | Hungarian matching |
| 속도 | 매우 빠름 | 느림 |
| 장거리 관계 | 어려움 | Self-attention으로 처리 |

---

## 2. YOLO 아키텍처 파이프라인 (★★★)

```
입력 이미지 [B, C, H, W]
        ↓
Backbone (ResNet18)
→ feature map: [B, 512, H/32, W/32]
        ↓
Detection Head (Conv layers)
→ raw output: [B, S, S, B*(5+C)]
  S=그리드 크기, B=셀당 bbox 수, C=클래스 수
        ↓
각 그리드 셀마다 예측:
  - bbox 좌표: (tx, ty, tw, th)  → sigmoid/exp 변환으로 실제 좌표
  - objectness: 물체가 있을 확률
  - class prob: 각 클래스 확률 (C개)
```

### 2.1 그리드 셀 예측 구조

이미지를 S×S 그리드로 분할. 각 셀은 B개의 bbox를 예측.

```
각 bbox 예측값 (5 + C):
├── tx, ty: bbox 중심의 셀 내 상대 좌표 (sigmoid → [0,1])
├── tw, th: bbox 크기 (exp → anchor 대비 비율)
├── confidence: objectness score (sigmoid)
└── c_1, ..., c_C: 클래스 확률 (softmax)
```

### 2.2 Anchor Box

각 그리드 셀이 사전 정의된 anchor 크기를 기준으로 bbox를 예측:

```python
# anchor는 (width, height) 쌍으로 미리 정의
# 예측값은 anchor 대비 offset/scale
bw = anchor_w * exp(tw)
bh = anchor_h * exp(th)
```

---

## 3. YOLO Loss 함수 (★★★ 구현 핵심)

### 3.1 매칭 방식 (DETR과의 차이)

DETR: Hungarian algorithm으로 최적 매칭
YOLO: **GT box와 가장 IoU가 높은 anchor에 직접 할당** (greedy)

```python
# GT bbox → 어느 그리드 셀? → 어느 anchor?
# IoU 가장 높은 anchor가 담당
best_anchor = argmax(IoU(gt_box, anchors))
```

### 3.2 3가지 Loss

**① Localization Loss**
```
담당 anchor (responsible)의 bbox 좌표 loss
L_loc = λ_coord * Σ [(tx-tx*)² + (ty-ty*)² + (tw-tw*)² + (th-th*)²]
```

**② Confidence Loss**
```
물체 있는 셀: confidence → 1 (실제 IoU 값)
물체 없는 셀: confidence → 0 (λ_noobj로 down-weight)

L_conf = Σ_obj (conf - IoU)² + λ_noobj * Σ_noobj (conf)²
```

`λ_noobj`: 배경 셀이 압도적으로 많으므로 가중치를 낮춤 (보통 0.5)

**③ Classification Loss**
```
담당 anchor 위치의 클래스 예측 loss
L_cls = Σ_obj CE(pred_class, gt_class)
또는 MSE 사용 (원 논문)
```

**최종:**
```
L = L_loc + L_conf + L_cls
```

---

## 4. NMS (Non-Maximum Suppression) (★★)

YOLO는 같은 물체에 대해 여러 셀이 중복 예측할 수 있음 → NMS로 제거

```
1. confidence 임계값 이하 박스 제거
2. confidence 높은 순서로 정렬
3. 남은 박스들과 IoU > threshold면 억제 (제거)
4. 반복
```

DETR은 Hungarian matching으로 중복이 원천 차단되어 NMS가 불필요.

---

## 5. 평가 지표 — mAP (★★)

### 5.1 AP (Average Precision)

클래스별 Precision-Recall 곡선의 넓이.

```
Precision = TP / (TP + FP)   # 예측한 것 중 맞은 비율
Recall    = TP / (TP + FN)   # 실제 있는 것 중 찾은 비율

AP = ∫ Precision d(Recall)   # P-R 곡선 아래 넓이
```

### 5.2 mAP (mean AP)

모든 클래스의 AP 평균:
```
mAP = (1/C) * Σ AP_c
```

VOC 2007: IoU ≥ 0.5 기준, 20개 클래스
- YOLO 목표: **mAP ≥ 0.53**
- DETR 목표: **mAP ≥ 0.58**

---

## 6. 훈련 설정 (★★)

### 6.1 YOLO 훈련 특징

```python
# Optimizer
SGD(lr=1e-3, momentum=0.9, weight_decay=5e-4)

# Scheduler
CosineAnnealingLR 또는 StepLR

# 배치 크기
batch_size = 16 또는 32  (DETR보다 큼, 모델이 가벼움)

# Backbone
ResNet18 (ImageNet pretrained)
```

### 6.2 DETR 대비 훈련 장점

- 훨씬 빠른 수렴 (anchor 기반이라 초기 학습 안정적)
- 더 큰 batch size 가능
- 메모리 효율 좋음

---

## 7. YOLO 구현 핵심 포인트 (★★)

### 7.1 좌표 변환

```python
# 예측값 → 실제 좌표
bx = sigmoid(tx) + cx   # cx: 그리드 셀 x 오프셋
by = sigmoid(ty) + cy
bw = anchor_w * exp(tw)
bh = anchor_h * exp(th)
```

### 7.2 Responsible anchor 선택

```python
# 각 GT box에 대해 IoU 가장 높은 anchor 선택
ious = compute_iou(gt_boxes, anchors)  # [N_gt, N_anchors]
best_anchor_idx = ious.argmax(dim=1)   # [N_gt]
```

### 7.3 λ 가중치

| 항목 | 가중치 | 이유 |
|---|---|---|
| λ_coord | 5.0 | 좌표 정확도 강조 |
| λ_noobj | 0.5 | 배경 셀 down-weight |
| λ_obj | 1.0 | 물체 있는 셀 |

---

## 8. YOLO vs DETR 핵심 비교 (★)

| 측면 | YOLO | DETR |
|---|---|---|
| **예측 구조** | S×S 그리드, B anchors/셀 | N개 object query |
| **GT 매칭** | IoU 기반 greedy | Hungarian (최적) |
| **중복 제거** | NMS 필요 | 불필요 |
| **장거리 관계** | 어려움 | Transformer self-attention |
| **속도** | 매우 빠름 (실시간 가능) | 느림 |
| **구현 복잡도** | anchor 설계 필요 | end-to-end 단순 |
| **Loss** | MSE/BCE | CE + L1 + GIoU |
| **Backbone** | ResNet18 | DINOv2 (ViT) |
| **mAP 목표** | 0.53 | 0.58 |

---

## 9. 제출 체크리스트

```
□ yolo_loss.py 구현 완료
□ Colab GPU로 15 epoch 훈련
□ mAP >= 0.53 확인
□ my_solution_yolo.csv 생성
□ Kaggle YOLO competition 제출 (Submit 버튼 클릭)
□ mp4a_yolo_output.pdf (노트북 출력 포함 PDF)
```
