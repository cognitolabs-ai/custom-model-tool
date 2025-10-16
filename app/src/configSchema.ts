import { z } from 'zod';

export const loraSchema = z.object({
  r: z.coerce.number().int().min(1).default(16),
  alpha: z.coerce.number().min(0).default(32),
  dropout: z.coerce.number().min(0).max(1).default(0.05),
  target_modules: z.array(z.string()).optional(),
  bias: z.enum(['none', 'all', 'lora_only']).default('none')
});

export const qloraSchema = z.object({
  bnb_4bit_quant_type: z.enum(['nf4', 'fp4']).default('nf4'),
  bnb_4bit_use_double_quant: z.coerce.boolean().default(true),
  bnb_4bit_compute_dtype: z.enum(['float16', 'float32', 'bfloat16']).default('bfloat16')
});

export const hrmSchema = z
  .object({
    enabled: z.boolean().default(false),
    max_steps: z.coerce.number().int().min(1).max(16).default(4),
    act: z
      .object({
        enabled: z.boolean().default(true),
        threshold: z.coerce.number().min(0.3).max(0.9).default(0.6),
        penalty: z.coerce.number().min(0).max(0.05).default(0.02)
      })
      .default({ enabled: true, threshold: 0.6, penalty: 0.02 }),
    halting_head: z
      .object({
        dim: z.coerce.number().int().min(16).max(256).default(64),
        init_bias: z.coerce.number().min(0).max(3).default(1.5)
      })
      .optional(),
    recursion: z
      .object({
        mode: z.enum(['fixed_point', 'recurrent']).default('fixed_point'),
        one_step_grad: z.boolean().default(true),
        deep_supervision: z.coerce.number().min(0).max(0.6).default(0.4),
        step_loss_weighting: z.enum(['uniform', 'cosine', 'geometric']).default('cosine')
      })
      .default({
        mode: 'fixed_point',
        one_step_grad: true,
        deep_supervision: 0.4,
        step_loss_weighting: 'cosine'
      }),
    controller: z
      .object({
        type: z.enum(['tiny_mlp', 'gru', 'tiny_transformer']).default('tiny_mlp'),
        hidden: z.coerce.number().int().min(32).max(256).default(128),
        layers: z.coerce.number().int().min(1).max(3).default(2),
        dropout: z.coerce.number().min(0).max(0.3).default(0.1)
      })
      .optional(),
    objective: z
      .object({
        task: z.enum(['classification', 'generation']).default('classification'),
        aux: z
          .array(z.enum(['consistency', 'entropy_min', 'margin']))
          .default(['consistency'])
          .optional()
      })
      .optional(),
    eval: z
      .object({
        compute_vs_quality_curve: z.boolean().default(true),
        budgets: z
          .array(z.coerce.number().int().min(1).max(16))
          .default([1, 2, 4, 8])
      })
      .optional()
  })
  .partial()
  .superRefine((data, ctx) => {
    if (data.enabled) {
      if (typeof data.max_steps !== 'number') {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'max_steps is required when HRM is enabled',
          path: ['max_steps']
        });
      }
      if (!data.act?.threshold) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'ACT threshold is required when HRM is enabled',
          path: ['act', 'threshold']
        });
      }
      if (!data.act?.penalty && data.act?.penalty !== 0) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'ACT penalty is required when HRM is enabled',
          path: ['act', 'penalty']
        });
      }
      if (!data.recursion?.one_step_grad) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'One-step gradient approximation should remain enabled for HRM stability',
          path: ['recursion', 'one_step_grad']
        });
      }
    }
  });

export const trainingModes = ['supervised', 'grpo', 'dpo', 'orpo'] as const;
export type TrainingMode = (typeof trainingModes)[number];

const datasetSplitSchema = z
  .object({
    train: z.coerce.number().min(0).max(1).optional(),
    val: z.coerce.number().min(0).max(1).optional(),
    test: z.coerce.number().min(0).max(1).optional()
  })
  .refine(
    (val) =>
      ['train', 'val', 'test'].some((key) => typeof val[key as keyof typeof val] === 'number'),
    'Provide at least one split ratio'
  )
  .optional();

export const datasetSchema = z
  .object({
    source: z.enum(['upload_local_path', 'hf_dataset_id']),
    id: z.string().optional(),
    path: z.string().optional(),
    format: z.enum(['jsonl_chat', 'jsonl_instr', 'csv_classification']),
    split: datasetSplitSchema
  })
  .superRefine((data, ctx) => {
    if (data.source === 'hf_dataset_id' && !data.id) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'HF dataset id is required when source is hf_dataset_id',
        path: ['id']
      });
    }
    if (data.source === 'upload_local_path' && !data.path) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'File path is required when source is upload_local_path',
        path: ['path']
      });
    }
  });

export const rlSchema = z
  .object({
    reward_model: z.string().min(1).optional(),
    reference_model: z.string().min(1).optional(),
    rollout_prompts_path: z.string().min(1).optional(),
    num_generations: z.coerce.number().int().min(1).optional(),
    beta: z.coerce.number().min(0).optional(),
    target_kl: z.coerce.number().min(0).optional(),
    max_steps: z.coerce.number().int().min(1).optional(),
    preference_dataset: z.string().min(1).optional(),
    warmup_steps: z.coerce.number().int().min(0).optional()
  })
  .optional();

export const compileSchema = z
  .object({
    enabled: z.boolean().default(false),
    backend: z.enum(['inductor', 'onnxrt', 'cudagraphs', 'eager']).default('inductor'),
    mode: z.enum(['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']).default('default'),
    fullgraph: z.boolean().default(false),
    dynamic: z.coerce.boolean().optional()
  })
  .optional();

export const profilerSchema = z
  .object({
    enabled: z.boolean().default(false),
    activities: z.array(z.enum(['cpu', 'cuda', 'xpu'])).nonempty().default(['cpu']),
    record_shapes: z.boolean().default(false),
    profile_memory: z.boolean().default(false),
    with_stack: z.boolean().default(false),
    with_flops: z.boolean().default(false),
    schedule_wait: z.coerce.number().int().min(0).default(1),
    schedule_warmup: z.coerce.number().int().min(0).default(1),
    schedule_active: z.coerce.number().int().min(0).default(3),
    tensorboard_trace_dir: z.string().optional()
  })
  .optional();

export const hyperparamsSchema = z.object({
  learning_rate: z.coerce.number().positive(),
  batch_size_train: z.coerce.number().int().min(1),
  batch_size_eval: z.coerce.number().int().min(1).optional(),
  num_epochs: z.coerce.number().int().min(1),
  gradient_accumulation: z.coerce.number().int().min(1),
  weight_decay: z.coerce.number().min(0).optional(),
  lr_scheduler: z.string().optional(),
  warmup_ratio: z.coerce.number().min(0).max(1),
  max_seq_len: z.coerce.number().int().min(1)
});

export const configSchema = z
  .object({
    provider: z.enum(['hf_local', 'hf_hub']),
    model_id: z.string().min(1),
    tune_type: z.enum(['full', 'lora', 'qlora']),
    accelerator: z.enum(['baseline', 'unsloth']).default('baseline'),
    training_mode: z.enum(trainingModes),
    seed: z.coerce.number().int().min(0).max(4_294_967_295).optional(),
    hyperparams: hyperparamsSchema,
    dataset: datasetSchema,
    eval: z.object({
      metrics: z.array(z.enum(['perplexity', 'rouge', 'accuracy', 'f1'])).nonempty()
    }),
    logs: z.enum(['tensorboard', 'none']),
    hw: z.object({
      device: z.enum(['auto', 'cpu', 'cuda']),
      mixed_precision: z.enum(['fp16', 'bf16', 'none'])
    }),
    artifacts: z.object({
      output_dir: z.string().min(1),
      push_to_hub: z.coerce.boolean().optional(),
      save_strategy: z.string().optional(),
      save_total_limit: z.coerce.number().int().min(1).optional()
    }),
    lora: loraSchema.optional(),
    qlora: qloraSchema.optional(),
    hrm: hrmSchema.optional(),
    rl: rlSchema,
    compile: compileSchema,
    profiler: profilerSchema
  })
  .superRefine((data, ctx) => {
    if (data.tune_type === 'lora' && !data.lora) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'LoRA settings are required when tune type is lora',
        path: ['lora']
      });
    }
    if (data.tune_type === 'qlora' && !data.qlora) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'QLoRA settings are required when tune type is qlora',
        path: ['qlora']
      });
    }
    if (data.hrm?.enabled) {
      if (data.hw.device !== 'cuda') {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'HRM is experimental and currently expects CUDA device',
          path: ['hw', 'device']
        });
      }
      if (data.hw.mixed_precision !== 'bf16' && data.hw.mixed_precision !== 'fp16') {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'HRM requires fp16 or bf16 mixed precision',
          path: ['hw', 'mixed_precision']
        });
      }
    }
    if (data.training_mode !== 'supervised' && !data.rl) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'RL settings are required for non-supervised training modes',
        path: ['rl']
      });
    }
    if (data.training_mode === 'grpo' && data.rl) {
      if (!data.rl.reward_model) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Reward model is required for GRPO',
          path: ['rl', 'reward_model']
        });
      }
      if (!data.rl.rollout_prompts_path) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Rollout prompts path is required for GRPO',
          path: ['rl', 'rollout_prompts_path']
        });
      }
    }
    if ((data.training_mode === 'dpo' || data.training_mode === 'orpo') && data.rl) {
      if (!data.rl.preference_dataset) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Preference dataset is required',
          path: ['rl', 'preference_dataset']
        });
      }
      if (!data.rl.reference_model) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Reference model is required',
          path: ['rl', 'reference_model']
        });
      }
    }
  });

export type UiConfig = z.infer<typeof configSchema>;
export type HrmConfig = NonNullable<UiConfig['hrm']>;
