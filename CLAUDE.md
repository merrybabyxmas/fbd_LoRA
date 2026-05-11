# FBD-LoRA 실험 설정 노트

## 환경

- Python venv: `/home/dongwoo39/.venv/bin/activate`
- LD_LIBRARY_PATH 필수: `export LD_LIBRARY_PATH=/home/dongwoo39/.venv/lib/python3.13/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH`
- rclone: `/home/dongwoo39/bin/rclone` (PATH에 없음, 절대경로 사용)
- GDrive remote: `fbd_gdrive` (`~/.config/rclone/rclone.conf` 설정 완료)
- Multi-GPU: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` 항상 설정

---

## 논문용 실험 세팅 (확정)

핵심 원칙: **baseline fairness** — 메소드 간에 가능한 한 동일한 rank/batch/lr/target_modules를 사용하고, 메소드 특성상 반드시 달라야 하는 것만 다르게 둔다.

---

## NLG 최종 세팅

### 공통 세팅 (모든 메소드 동일)

| 항목 | 값 | 근거 |
|------|-----|------|
| model | Mistral-7B-v0.1 | — |
| dataset | fxmeng/pissa-dataset | PiSSA 논문과 동일 데이터셋 |
| max_samples | 100,000 | PiSSA 논문 동일 |
| max_seq_length | 2048 | PiSSA 논문 동일 |
| epochs | 1 | PiSSA 논문 동일 |
| per_device_batch | 4 | — |
| grad_accum | 8 | — |
| **global batch (4 GPU)** | **128** | 4 × 8 × 4 = 128 → PiSSA 논문(batch=128)과 일치 |
| **lr** | **5e-5** | 아래 근거 참조 |
| lr_scheduler | cosine | PiSSA 논문 동일 |
| warmup_ratio | 0.03 | PiSSA 논문 동일 |
| weight_decay | 0.0 | PiSSA 논문 동일 |
| rank | 16 | — |
| alpha | 16 | PiSSA 논문: `lora_alpha = lora_r` |
| **dropout** | **0.0** | PiSSA 논문: `lora_dropout = 0` |
| target_modules | q_proj, k_proj, v_proj, o_proj | Main; all-linear은 appendix |
| precision | bf16 | — |

**lr=5e-5 근거:**
- PiSSA 논문(arXiv:2404.02948)은 `lr=2e-5`를 사용
- `lr=2e-4`는 일반 LoRA 설정으로는 흔하나 PiSSA protocol과는 10배 차이
- 중간값 `5e-5`를 main으로 설정하고, appendix에서 `{2e-5, 5e-5, 1e-4, 2e-4}` sweep
- LoRA learning rate sensitivity에 관해: "Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning" — https://www.researchgate.net/publication/400505506

**dropout=0.0 근거:**
- PiSSA 논문(arXiv:2404.02948) 명시: `lora_dropout = 0`
- FBD의 gradient routing 자체가 training dynamics를 수정하므로, dropout이 있으면 분석이 혼탁해짐
- 출처: https://arxiv.org/pdf/2404.02948

> ⚠️ 수정 이력 (2026-05-11): `lr: 2e-4 → 5e-5`, `dropout: 0.05 → 0.0`

### 메소드별 차이 (method-specific만)

| Method | 추가 설정 | 비고 |
|--------|-----------|------|
| LoRA | — | baseline |
| DoRA | `use_dora: true` | PEFT LoraConfig |
| PiSSA | `init_lora_weights: pissa_niter_16` | PEFT LoraConfig |
| AdaLoRA | `init_r: 24, target_r: 16` | 최종 rank budget = 16으로 맞춤 |
| FBD-LoRA | `fbd.enabled: true` + pullback metric | 내부 구현 |

> ⚠️ 수정 이력 (2026-05-11): AdaLoRA `target_r: 8 → 16, init_r: 12 → 24`.
> 기존 설정은 다른 메소드 rank=16 대비 파라미터 수가 절반이어서 불공정했음.

### 추가 실험 (Ablation / Appendix)

| 실험 | 세팅 |
|------|------|
| LR sweep | `{2e-5, 5e-5, 1e-4, 2e-4}` |
| Rank sweep | `{4, 8, 16, 32}` (LoRA, PiSSA, FBD만) |
| Target modules | attention-only vs all-linear (gate_proj, up_proj, down_proj 추가) |
| FBD lambda | `{0.25, 0.5, 0.75}` |
| FBD routing | true_grad / direct_w0x / pullback / pullback+gate |

### 논문 기술 문장 (실험 섹션용)

> For NLG experiments, we follow the PiSSA protocol using Mistral-7B-v0.1 and the publicly released PiSSA processed dataset. Unless otherwise stated, all methods are trained for one epoch on 100K examples with sequence length 2048, AdamW optimizer, cosine scheduling, warmup ratio 0.03, and no weight decay. We use rank 16 with α=16 and dropout 0 for all LoRA-style methods, adapting the q, k, v, and o projections. On four GPUs, our per-device batch size of 4 and gradient accumulation of 8 gives a global batch size of 128, matching the PiSSA training protocol.

### 참조 논문

- PiSSA (NeurIPS 2024 Spotlight): https://arxiv.org/abs/2404.02948
- LoRA (Hu et al., 2021): https://arxiv.org/abs/2106.09685
- MetaMath (Yu et al., 2023): https://arxiv.org/abs/2309.12284
- PiSSA dataset: https://huggingface.co/datasets/fxmeng/pissa-dataset

---

## Image 최종 세팅

### 공통 세팅

| 항목 | 값 |
|------|----|
| model | runwayml/stable-diffusion-v1-5 |
| dataset | DreamBench++ (yuangpeng/dreambench_plus, official files) |
| resolution | 512 |
| max_steps | 1000 (equal-budget comparison) |
| precision | fp16 |
| num_images_per_prompt | 4 |
| num_inference_steps | 50 |
| guidance_scale | 7.5 |
| seed | 42 |
| metrics | CLIP-I, CLIP-T, DINO |

### 메소드별 세팅

| Method | lr | batch | grad_accum | eff_batch | rank | alpha | 비고 |
|--------|-----|-------|-----------|-----------|------|-------|------|
| FBD-LoRA | 1e-4 | 1 | 4 | 4 | 8 | 8 | to_q,k,v,out |
| LoRA | 1e-4 | 1 | 4 | 4 | 8 | 8 | to_q,k,v,out |
| DreamBooth | 5e-6 | 1 | 4 | 4 | — | — | 전체 UNet |
| **Custom Diffusion** | **8e-5** | 2 | 2 | 4 | — | — | K,V + concept token |
| SVDiff | — | — | — | — | — | — | EXTERNAL_REQUIRED |
| DiffuseKronA | — | — | — | — | — | — | EXTERNAL_REQUIRED |

**FBD-LoRA / LoRA lr=1e-4 근거:**
- HuggingFace Diffusers 공식 DreamBooth+LoRA 학습 가이드 표준값
- 출처: https://huggingface.co/docs/diffusers/training/dreambooth

**DreamBooth lr=5e-6 근거:**
- HuggingFace Diffusers 공식 DreamBooth 학습 스크립트 기본값
- 출처: https://huggingface.co/docs/diffusers/v0.11.0/en/training/dreambooth
- 해당 구절: `accelerate launch train_dreambooth.py --learning_rate=5e-6`
- 문서 명시: "Dreambooth fine-tuning is very sensitive to hyperparameters and easy to overfit."
- 논문: Ruiz et al., 2022 — https://arxiv.org/abs/2208.12242
- LoRA(1e-4)와 다른 이유: 전체 UNet 학습이므로 동일 lr 사용 시 발산

> ⚠️ 수정 이력 (2026-05-11): Custom Diffusion `lr: 1e-5 → 8e-5`

**Custom Diffusion lr=8e-5 근거:**
- Custom Diffusion 원 논문(CVPR 2023) 학습 설정: `lr=8e-5, batch=8, single concept 250 steps`
- 출처: https://openaccess.thecvf.com/content/CVPR2023/papers/Kumari_Multi-Concept_Customization_of_Text-to-Image_Diffusion_CVPR_2023_paper.pdf
- 공식 레포: https://github.com/adobe-research/custom-diffusion
- 기존 `lr=1e-5`는 Diffusers 구현 기본값이나 원 논문 대비 지나치게 보수적 → Custom Diffusion baseline이 약하게 나올 위험
- 논문: Kumari et al., CVPR 2023 — https://arxiv.org/abs/2212.04488
- Equal-budget 비교 기준: 모든 메소드 1000 steps로 통일 (original 250 steps는 appendix에 기재)

### 추가 실험 (Appendix)

| 실험 | 세팅 |
|------|------|
| Rank sweep | `{4, 8, 16}` (LoRA, FBD만) |
| Step sweep | Custom Diffusion original: `lr=8e-5, steps=250, eff_batch=8` |
| DreamBooth trainable params 비교 | full UNet vs LoRA adapter 파라미터/용량 표 |

### 논문 기술 문장 (실험 섹션용)

> For image personalization, we use Stable Diffusion v1.5 and evaluate on DreamBench++ official files. LoRA-DreamBooth and FBD-LoRA use rank 8, α=8, learning rate 1e-4, effective batch size 4, and 1000 training steps, adapting the cross-attention projections. DreamBooth full fine-tuning uses learning rate 5e-6 following the official HuggingFace implementation. Custom Diffusion uses learning rate 8e-5 following the original paper. All methods use the same training budget of 1000 steps for fair comparison. We generate four images per prompt with 50 denoising steps and guidance scale 7.5, and report CLIP-I and DINO for subject fidelity and CLIP-T for prompt alignment.

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

# 본 실험 NLG (4 GPU, global batch 128)
bash scripts/nlg/metamath/train_fbd.sh 0,1,2,3 true
bash scripts/nlg/metamath/train_lora.sh 0,1,2,3 true

# GSM8K eval
bash experiments/nlg/metamath/eval_gsm8k.sh <checkpoint_path>
```
