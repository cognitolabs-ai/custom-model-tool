# Configuration Reference

The generator consumes a single JSON document validated against `schemas/config.schema.json`. This guide summarises supported fields.

## Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string (`hf_local` \| `hf_hub`) | ✓ | Where the base model checkpoint lives. |
| `model_id` | string | ✓ | Hugging Face model ID or local path. |
| `tune_type` | string (`full` \| `lora` \| `qlora`) | ✓ | Training strategy; enables additional blocks below. |
| `accelerator` | string (`baseline` \| `unsloth`) | ✗ (default `baseline`) | Toggle Unsloth fast-path for LoRA/QLoRA runs. |
| `training_mode` | string (`supervised` \| `grpo` \| `dpo` \| `orpo`) | ✗ (default `supervised`) | Select RL fine-tuning strategy. Non-supervised modes require the `rl` block. |
| `seed` | integer | ✗ | Deterministic seed (0 – 2^32-1). |
| `hyperparams` | object | ✓ | Core training hyperparameters (see below). |
| `dataset` | object | ✓ | Dataset source, format, and splits. |
| `eval` | object | ✓ | Evaluation metrics (`perplexity`, `rouge`, `accuracy`, `f1`). |
| `logs` | string (`tensorboard` \| `none`) | ✓ | Logging backend during training. |
| `hw` | object | ✓ | Hardware preferences (`device`, `mixed_precision`). |
| `artifacts` | object | ✓ | Output directory + optional checkpoint behaviour. |
| `lora` | object | Required when `tune_type = "lora"` | LoRA adapter hyperparameters. |
| `qlora` | object | Required when `tune_type = "qlora"` | 4-bit quantisation options. |
| `rl` | object | Required when `training_mode` is `grpo`, `dpo`, or `orpo` | Reinforcement learning settings (details below). |
| `compile` | object | ✗ | `torch.compile` options. |
| `profiler` | object | ✗ | `torch.profiler` instrumentation settings. |
| `hrm` | object | ✗ | Hyper-Recursive Module overlay toggles (experimental). |

## Hyperparameters (`hyperparams`)

| Field | Type | Notes |
|-------|------|-------|
| `learning_rate` | number > 0 | Base learning rate. |
| `batch_size_train` | integer ≥ 1 | Per-device batch size for training. |
| `batch_size_eval` | integer ≥ 1 | Optional; defaults to train batch size. |
| `num_epochs` | integer ≥ 1 | Total epochs. |
| `gradient_accumulation` | integer ≥ 1 | Accumulation steps. |
| `weight_decay` | number ≥ 0 | L2 regularisation (optional). |
| `lr_scheduler` | string | Trainer scheduler key (`cosine`, `linear`, etc.). |
| `warmup_ratio` | number 0–1 | LR warmup fraction. |
| `max_seq_len` | integer ≥ 1 | Max sequence length for tokenisation. |

## Dataset (`dataset`)

| Field | Type | Description |
|-------|------|-------------|
| `source` | `hf_dataset_id` \| `upload_local_path` | Choose remote or local dataset. |
| `id` | string | Required when `source = hf_dataset_id`; Hugging Face dataset ID. |
| `path` | string | Required when `source = upload_local_path`; local path to file. |
| `format` | `jsonl_chat` \| `jsonl_instr` \| `csv_classification` | Determines tokenisation strategy. |
| `split.train` / `split.val` / `split.test` | number 0–1 | Optional split ratios. |

## Reinforcement Learning (`rl`)

| Training Mode | Required Fields | Optional Fields |
|---------------|-----------------|-----------------|
| `grpo` | `reward_model`, `rollout_prompts_path` | `num_generations`, `target_kl`, `max_steps`, `beta`, `warmup_steps` |
| `dpo` | `preference_dataset`, `reference_model` | `beta`, `warmup_steps`, `max_steps` |
| `orpo` | `preference_dataset`, `reference_model` | `beta`, `target_kl`, `warmup_steps`, `max_steps` |

`reference_model` should point to a baseline policy checkpoint used for divergence penalties. `preference_dataset` expects pairwise preference data (chosen/rejected pairs).

## LoRA (`lora`)

| Field | Type | Description |
|-------|------|-------------|
| `r` | integer ≥ 1 | Rank of adapter matrices. |
| `alpha` | number ≥ 0 | Scaling factor. |
| `dropout` | number 0–1 | Dropout applied to LoRA adapters. |
| `target_modules` | string[] | Optional module names receiving adapters. |
| `bias` | `none` \| `all` \| `lora_only` | Bias handling strategy. |

## QLoRA (`qlora`)

| Field | Type | Description |
|-------|------|-------------|
| `bnb_4bit_quant_type` | `nf4` \| `fp4` | 4-bit quantisation flavour. |
| `bnb_4bit_use_double_quant` | boolean | Enable double quantisation. |
| `bnb_4bit_compute_dtype` | `float16` \| `float32` \| `bfloat16` | Compute dtype. |

## torch.compile (`compile`)

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Wrap model with `torch.compile`. |
| `backend` | `inductor` \| `onnxrt` \| `cudagraphs` \| `eager` | Backend passed to PyTorch. |
| `mode` | `default` \| `reduce-overhead` \| `max-autotune` \| `max-autotune-no-cudagraphs` | Optimisation profile. |
| `fullgraph` | boolean | Require single-graph capture. |
| `dynamic` | boolean | Prefer dynamic shapes. |

## torch.profiler (`profiler`)

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Enable profiling context. |
| `activities` | array of `cpu` \| `cuda` \| `xpu` | Devices to profile. |
| `record_shapes` | boolean | Capture tensor shapes. |
| `profile_memory` | boolean | Track memory allocations. |
| `with_stack` | boolean | Include stack traces. |
| `with_flops` | boolean | Estimate FLOPs. |
| `schedule_wait` / `schedule_warmup` / `schedule_active` | integer ≥ 0 | Profiler schedule. |
| `tensorboard_trace_dir` | string | Optional output directory for traces. |

## HRM Overlay (`hrm`)

Experimental adaptive-computation controller options. When `enabled` is true, ensure `hw.device = "cuda"` and `hw.mixed_precision` is `fp16` or `bf16`.

## Examples

See the `examples/configs/` directory for ready-to-use configurations demonstrating full fine-tuning, LoRA, QLoRA, and RL presets.
