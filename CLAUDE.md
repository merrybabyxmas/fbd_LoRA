# FBD-LoRA 실험 설정 노트

## 환경

- Python venv: `/home/dongwoo39/.venv/bin/activate`
- LD_LIBRARY_PATH 필수: `export LD_LIBRARY_PATH=/home/dongwoo39/.venv/lib/python3.13/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH`
- rclone: `/home/dongwoo39/bin/rclone` (PATH에 없음, 절대경로 사용)
- GDrive remote: `fbd_gdrive` (`~/.config/rclone/rclone.conf` 설정 완료)
- Multi-GPU: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` 항상 설정

---

## Hyperparameter 설정 근거

### NLG (Mistral-7B-v0.1, pissa-dataset)

모든 메소드 공통:
| 항목 | 값 |
|------|----|
| lr | 2e-4 |
| batch_size | 4 |
| grad_accum | 8 (eff. batch 32) |
| epochs | 1 |
| warmup_ratio | 0.03 |
| lr_scheduler | cosine |
| rank | 16 |
| alpha | 16 |
| dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj |

**AdaLoRA 전용:**
| 항목 | 값 | 근거 |
|------|----|------|
| init_r | 24 | target_r의 1.5배 (AdaLoRA 논문 권장: 최종 rank보다 크게 시작) |
| target_r | 16 | 다른 메소드 rank=16과 동일 (공정 비교) |
| alpha | 16 | 다른 메소드와 동일 |

> ⚠️ 수정 이력: 최초 `target_r: 8, init_r: 12`로 잘못 설정되어 있었음 (2026-05-11 수정).
> 다른 메소드 rank=16 대비 파라미터 수가 절반이어서 공정한 비교 불가였음.

---

### Image (Stable Diffusion v1.5, DreamBench++)

#### LoRA / FBD-LoRA
| 항목 | 값 |
|------|----|
| lr | 1e-4 |
| batch_size | 1 |
| rank | 8 |
| alpha | 8 |
| dropout | 0.0 |
| target_modules | to_q, to_k, to_v, to_out.0 |

#### DreamBooth (full UNet fine-tuning)
| 항목 | 값 | 출처 |
|------|----|------|
| lr | **5e-6** | HuggingFace Diffusers 공식 DreamBooth 학습 스크립트 기본값 |
| batch_size | 1 | |

**출처:**
- URL: https://huggingface.co/docs/diffusers/v0.11.0/en/training/dreambooth
- 해당 구절:
  ```bash
  accelerate launch train_dreambooth.py \
    --learning_rate=5e-6 \
  ```
- 추가 언급: "Dreambooth fine-tuning is very sensitive to hyperparameters and easy to overfit."
- 논문: Ruiz et al., 2022 — https://arxiv.org/abs/2208.12242

> LoRA(1e-4)와 다른 이유: DreamBooth는 전체 UNet을 학습하므로 파라미터 수가 훨씬 많아 동일 lr 사용 시 발산함.

#### Custom Diffusion
| 항목 | 값 | 출처 |
|------|----|------|
| lr | **1e-5** | Adobe Research 공식 Custom Diffusion 레포 기본값 |
| batch_size | 2 | 공식 레포 기본값 |

**출처:**
- URL: https://github.com/adobe-research/custom-diffusion
- 해당 구절 (공식 학습 스크립트 인자):
  ```
  --learning_rate=1e-5
  --lr_warmup_steps=0
  --max_train_steps=250
  ```
- 논문: Kumari et al., CVPR 2023 — https://arxiv.org/abs/2212.04488
- CVPR paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf

> 참고: 사람 얼굴의 경우 5e-6 + max_train_steps=750 권장 (동일 출처).

---

## GDrive 설정

- 체크포인트 저장 → `rclone copy` 업로드 → `rclone check` 무결성 검증 → 로컬 자동 삭제
- 업로드 경로: `fbd_gdrive:FBD_LORA_EXPERIMENTS/<run_id>/checkpoints/<label>/`
- 업로드 실패 또는 검증 실패 시 로컬 보존 (절대 삭제 안 함)
- 학습 로그 (`logs/`), eval 결과 (`eval/`)는 GDrive 대상 아님 — 로컬 유지
- 실험 configs (`configs/nlg/`, `configs/imagen/`): `upload_to_gdrive: true`
- Sanity configs (`configs/sanity/`): `upload_to_gdrive: false`

---

## 출력 디렉토리 구조

```
outputs/runs/<run_id>/
├── config.yaml             ← 실험 config 복사본
├── metadata.json           ← run_id, 시각, 메소드 정보
├── checkpoints/
│   ├── step_000010pct/     ← 10% 시점 (GDrive 업로드 후 로컬 삭제됨)
│   ├── step_000100pct/
│   └── final/
├── logs/
│   ├── train.log
│   └── wandb_id.txt
└── eval/
    └── gsm8k/
        ├── predictions.json
        └── metrics.json
```

---

## 주요 실험 스크립트

```bash
# NLG sanity (20-step, Mistral-7B)
bash scripts/sanity/nlg/metamath/train_{fbd,lora,dora,pissa,adalora}_mistral.sh

# Image sanity (20-step, SD1.5, DreamBench++)
bash scripts/sanity/imagen/dreambench/train_fbd_sd15_dreambench_plus.sh 1 false
bash scripts/sanity/imagen/dreambench/train_lora_sd15_dreambench_plus.sh 1 false

# 본 실험 NLG
bash scripts/nlg/metamath/train_fbd.sh 0,1,2 true
bash scripts/nlg/metamath/train_lora.sh 0,1,2 true

# GSM8K eval
bash experiments/nlg/metamath/eval_gsm8k.sh <checkpoint_path>
```
