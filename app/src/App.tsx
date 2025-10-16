
import { FormEvent, useEffect, useMemo, useState } from 'react';

import { renderBundle, type JsonObject } from '@core/renderBundle';

import {
  configSchema,
  type HrmConfig,
  type TrainingMode,
  type UiConfig
} from './configSchema';
import { InfoTooltip } from './InfoTooltip';

type Status = 'idle' | 'working' | 'error' | 'success';
type Metric = 'perplexity' | 'rouge' | 'accuracy' | 'f1';

interface DownloadLink {
  filename: string;
  url: string;
  size: string;
  description: string;
}

const DEFAULT_LORA: NonNullable<UiConfig['lora']> = {
  r: 16,
  alpha: 32,
  dropout: 0.05,
  target_modules: ['q_proj', 'v_proj'],
  bias: 'none'
};

const DEFAULT_QLORA: NonNullable<UiConfig['qlora']> = {
  bnb_4bit_quant_type: 'nf4',
  bnb_4bit_use_double_quant: true,
  bnb_4bit_compute_dtype: 'bfloat16'
};

const DEFAULT_COMPILE: NonNullable<UiConfig['compile']> = {
  enabled: false,
  backend: 'inductor',
  mode: 'default',
  fullgraph: false,
  dynamic: true
};

const DEFAULT_PROFILER: NonNullable<UiConfig['profiler']> = {
  enabled: false,
  activities: ['cpu'],
  record_shapes: false,
  profile_memory: false,
  with_stack: false,
  with_flops: false,
  schedule_wait: 1,
  schedule_warmup: 1,
  schedule_active: 3,
  tensorboard_trace_dir: './runs/profiler'
};

const DEFAULT_RL_TEMPLATES: Record<'grpo' | 'dpo' | 'orpo', NonNullable<UiConfig['rl']>> = {
  grpo: {
    reward_model: 'unsloth/reward-model',
    rollout_prompts_path: './prompts/sample_prompts.jsonl',
    num_generations: 4,
    target_kl: 6,
    max_steps: 256,
    warmup_steps: 50
  },
  dpo: {
    preference_dataset: 'your-org/dpo-preferences',
    reference_model: 'mistralai/Mistral-7B-Instruct-v0.3',
    beta: 0.1,
    warmup_steps: 50
  },
  orpo: {
    preference_dataset: 'your-org/orpo-preferences',
    reference_model: 'mistralai/Mistral-7B-Instruct-v0.3',
    beta: 0.1,
    target_kl: 3,
    warmup_steps: 50
  }
};

type HrmPresetKey = 'fast' | 'balanced' | 'quality';

const createHrmPreset = (overrides: Partial<HrmConfig> = {}): HrmConfig => ({
  enabled: overrides.enabled ?? true,
  max_steps: overrides.max_steps ?? 4,
  act: {
    enabled: overrides.act?.enabled ?? true,
    threshold: overrides.act?.threshold ?? 0.6,
    penalty: overrides.act?.penalty ?? 0.02
  },
  halting_head: {
    dim: overrides.halting_head?.dim ?? 128,
    init_bias: overrides.halting_head?.init_bias ?? 1.5
  },
  recursion: {
    mode: overrides.recursion?.mode ?? 'fixed_point',
    one_step_grad: overrides.recursion?.one_step_grad ?? true,
    deep_supervision: overrides.recursion?.deep_supervision ?? 0.4,
    step_loss_weighting: overrides.recursion?.step_loss_weighting ?? 'cosine'
  },
  controller: {
    type: overrides.controller?.type ?? 'tiny_mlp',
    hidden: overrides.controller?.hidden ?? 128,
    layers: overrides.controller?.layers ?? 2,
    dropout: overrides.controller?.dropout ?? 0.1
  },
  objective: {
    task: overrides.objective?.task ?? 'classification',
    aux: overrides.objective?.aux ?? ['consistency']
  },
  eval: {
    compute_vs_quality_curve: overrides.eval?.compute_vs_quality_curve ?? true,
    budgets: overrides.eval?.budgets ?? [1, 2, 4, 8]
  }
});

const HRM_PRESETS: Record<HrmPresetKey, HrmConfig> = {
  fast: createHrmPreset({
    max_steps: 2,
    act: { enabled: true, threshold: 0.75, penalty: 0.01 },
    recursion: { deep_supervision: 0.2, step_loss_weighting: 'uniform' },
    controller: { hidden: 64, layers: 1, type: 'tiny_mlp', dropout: 0.05 },
    objective: { task: 'classification', aux: ['consistency'] }
  }),
  balanced: createHrmPreset({}),
  quality: createHrmPreset({
    max_steps: 8,
    act: { enabled: true, threshold: 0.5, penalty: 0.03 },
    recursion: { deep_supervision: 0.5, step_loss_weighting: 'geometric' },
    controller: { hidden: 192, layers: 2, type: 'tiny_mlp', dropout: 0.1 },
    objective: { task: 'generation', aux: ['consistency', 'entropy_min'] }
  })
};

const HRM_PRESET_OPTIONS: Array<{ key: HrmPresetKey; label: string }> = [
  { key: 'fast', label: 'Fast' },
  { key: 'balanced', label: 'Balanced' },
  { key: 'quality', label: 'Max Quality' }
];

const METRICS: Metric[] = ['perplexity', 'rouge', 'accuracy', 'f1'];

const DEFAULT_CONFIG: UiConfig = {
  provider: 'hf_hub',
  model_id: 'mistralai/Mistral-7B-Instruct-v0.3',
  tune_type: 'full',
  accelerator: 'baseline',
  training_mode: 'supervised',
  seed: 42,
  hyperparams: {
    learning_rate: 0.0002,
    batch_size_train: 4,
    batch_size_eval: 4,
    num_epochs: 1,
    gradient_accumulation: 2,
    weight_decay: 0.01,
    lr_scheduler: 'cosine',
    warmup_ratio: 0.03,
    max_seq_len: 2048
  },
  dataset: {
    source: 'hf_dataset_id',
    id: 'samsum',
    format: 'jsonl_instr',
    split: {
      train: 0.98,
      val: 0.02
    }
  },
  eval: {
    metrics: ['rouge']
  },
  logs: 'tensorboard',
  hw: {
    device: 'auto',
    mixed_precision: 'bf16'
  },
  artifacts: {
    output_dir: 'outputs/run_full',
    save_strategy: 'epoch',
    save_total_limit: 2
  },
  compile: { ...DEFAULT_COMPILE },
  profiler: { ...DEFAULT_PROFILER },
  lora: undefined,
  qlora: undefined,
  rl: undefined,
  hrm: { enabled: false }
};

const splitInfo: Record<'train' | 'val' | 'test', string> = {
  train: 'Fraction of data used for training when auto-splitting (0-1).',
  val: 'Validation split ratio (0-1) for monitoring training quality.',
  test: 'Optional test split ratio held out for final evaluation (0-1).'
};

const formatBytes = (size: number): string => {
  if (size === 0) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  const exponent = Math.min(Math.floor(Math.log(size) / Math.log(1024)), units.length - 1);
  const value = size / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
};

const parseOrDefault = (raw: string, fallback: number): number => {
  const parsed = Number(raw);
  return Number.isNaN(parsed) ? fallback : parsed;
};

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

const ensureRl = (mode: TrainingMode, existing?: UiConfig['rl']): UiConfig['rl'] => {
  if (mode === 'supervised') {
    return undefined;
  }
  return clone(existing ?? DEFAULT_RL_TEMPLATES[mode]);
};

const normalizeHrm = (hrm?: UiConfig['hrm']): HrmConfig =>
  createHrmPreset({
    enabled: hrm?.enabled ?? true,
    max_steps: hrm?.max_steps,
    act: hrm?.act,
    halting_head: hrm?.halting_head,
    recursion: hrm?.recursion,
    controller: hrm?.controller,
    objective: hrm?.objective,
    eval: hrm?.eval
  });

const cloneHrm = (hrm: HrmConfig): HrmConfig => clone(hrm);
function App() {
  const [config, setConfig] = useState<UiConfig>({ ...DEFAULT_CONFIG });
  const [status, setStatus] = useState<Status>('idle');
  const [errors, setErrors] = useState<string[]>([]);
  const [downloads, setDownloads] = useState<DownloadLink[]>([]);

  useEffect(() => {
    return () => {
      downloads.forEach((item) => URL.revokeObjectURL(item.url));
    };
  }, [downloads]);

  const validationResult = useMemo(() => configSchema.safeParse(config), [config]);
  const previewConfig = useMemo(() => {
    const data = validationResult.success ? validationResult.data : config;
    return JSON.stringify(data, null, 2);
  }, [validationResult, config]);

  const isHrmEnabled = Boolean(config.hrm?.enabled);
  const hrmValues = normalizeHrm(isHrmEnabled ? config.hrm : undefined);
  const rlValues = config.training_mode === 'supervised' ? undefined : ensureRl(config.training_mode, config.rl);
  const compileState = config.compile ?? { ...DEFAULT_COMPILE };
  const profilerState = config.profiler ?? { ...DEFAULT_PROFILER };

  const cleanupDownloads = () => {
    downloads.forEach((item) => URL.revokeObjectURL(item.url));
    setDownloads([]);
  };

  const handleTuneTypeChange = (next: UiConfig['tune_type']) => {
    setConfig((prev) => {
      const nextConfig: UiConfig = {
        ...prev,
        tune_type: next
      };

      if (next === 'lora') {
        nextConfig.lora = prev.lora ?? clone(DEFAULT_LORA);
        nextConfig.qlora = undefined;
      } else if (next === 'qlora') {
        nextConfig.qlora = prev.qlora ?? clone(DEFAULT_QLORA);
        nextConfig.lora = undefined;
      } else {
        nextConfig.lora = undefined;
        nextConfig.qlora = undefined;
        if (prev.accelerator === 'unsloth') {
          nextConfig.accelerator = 'baseline';
        }
      }
      return nextConfig;
    });
  };

  const handleAcceleratorChange = (next: UiConfig['accelerator']) => {
    setConfig((prev) => {
      const nextConfig: UiConfig = { ...prev, accelerator: next };
      if (next === 'unsloth' && prev.tune_type === 'full') {
        nextConfig.tune_type = 'qlora';
        nextConfig.qlora = prev.qlora ?? clone(DEFAULT_QLORA);
        nextConfig.lora = undefined;
      }
      return nextConfig;
    });
  };

  const handleTrainingModeChange = (mode: TrainingMode) => {
    setConfig((prev) => ({
      ...prev,
      training_mode: mode,
      rl: ensureRl(mode, prev.rl)
    }));
  };

  const applyHrmPreset = (preset: HrmPresetKey) => {
    setConfig((prev) => ({
      ...prev,
      hrm: cloneHrm(HRM_PRESETS[preset])
    }));
  };

  const handleHrmToggle = (enabled: boolean) => {
    setConfig((prev) => {
      if (!enabled) {
        return { ...prev, hrm: { enabled: false } };
      }
      if (prev.hrm?.enabled) {
        return { ...prev, hrm: cloneHrm(normalizeHrm(prev.hrm)) };
      }
      return { ...prev, hrm: cloneHrm(HRM_PRESETS.balanced) };
    });
  };

  const updateHrm = (updater: (current: HrmConfig) => HrmConfig) => {
    setConfig((prev) => {
      if (!prev.hrm?.enabled) {
        return prev;
      }
      const next = updater(normalizeHrm(prev.hrm));
      return {
        ...prev,
        hrm: cloneHrm(next)
      };
    });
  };

  const updateRl = (updater: (current: NonNullable<UiConfig['rl']>) => NonNullable<UiConfig['rl']>) => {
    setConfig((prev) => {
      if (prev.training_mode === 'supervised') {
        return prev;
      }
      const base = ensureRl(prev.training_mode, prev.rl) ?? DEFAULT_RL_TEMPLATES.grpo;
      return {
        ...prev,
        rl: updater(clone(base))
      };
    });
  };

  const handleMetricToggle = (metric: Metric) => {
    setConfig((prev) => {
      const nextMetrics = new Set(prev.eval.metrics);
      if (nextMetrics.has(metric)) {
        if (nextMetrics.size === 1) {
          return prev;
        }
        nextMetrics.delete(metric);
      } else {
        nextMetrics.add(metric);
      }
      return {
        ...prev,
        eval: {
          ...prev.eval,
          metrics: Array.from(nextMetrics)
        }
      };
    });
  };

  const handleDatasetSplit = (key: 'train' | 'val' | 'test', raw: string) => {
    setConfig((prev) => {
      const nextSplit = { ...(prev.dataset.split ?? {}) };
      if (raw === '') {
        delete nextSplit[key];
      } else {
        const parsed = Number(raw);
        if (!Number.isNaN(parsed)) {
          nextSplit[key] = parsed;
        }
      }
      return {
        ...prev,
        dataset: {
          ...prev.dataset,
          split: Object.keys(nextSplit).length === 0 ? undefined : nextSplit
        }
      };
    });
  };

  const handleGenerate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setStatus('working');
    setErrors([]);

    const parsed = configSchema.safeParse(config);
    if (!parsed.success) {
      setStatus('error');
      setErrors(parsed.error.issues.map((issue) => `${issue.path.join('.') || 'config'}: ${issue.message}`));
      return;
    }

    try {
      const rendered = await renderBundle(parsed.data as unknown as JsonObject, { createZip: true });
      const nextDownloads: DownloadLink[] = [
        createDownloadLink(
          'fine_tune.ipynb',
          [rendered.notebookJson],
          'application/x-ipynb+json',
          'Notebook ready for execution in Jupyter or Colab.'
        ),
        createDownloadLink(
          'config.yaml',
          [rendered.configYaml],
          'text/yaml',
          'YAML configuration mirroring the chosen parameters.'
        ),
        createDownloadLink(
          'README.md',
          [rendered.readmeMarkdown],
          'text/markdown',
          'Companion README summarising the run.'
        )
      ];

      if (rendered.zipData) {
        nextDownloads.push(
          createDownloadLink(
            'notebook_bundle.zip',
            [rendered.zipData],
            'application/zip',
            'ZIP bundle containing the notebook, config, and README.'
          )
        );
      }

      cleanupDownloads();
      setDownloads(nextDownloads);
      setStatus('success');
    } catch (error) {
      setStatus('error');
      setErrors([(error as Error).message]);
    }
  };

  const resetForm = () => {
    cleanupDownloads();
    setConfig({ ...DEFAULT_CONFIG, compile: { ...DEFAULT_COMPILE }, profiler: { ...DEFAULT_PROFILER } });
    setErrors([]);
    setStatus('idle');
  };
  return (
    <div className="min-h-screen bg-slate-950">
      <header className="border-b border-slate-800 bg-slate-950/90">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-6 pb-10 pt-10 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-4">
              <h1 className="text-3xl font-semibold text-white">CognitioLabs Fine-Tuning Notebook Generator</h1>
              <div className="inline-flex items-center gap-3 text-sm text-slate-400">
                <a
                  className="rounded-full border border-brand-500/40 px-3 py-1 text-brand-200 transition hover:border-brand-400 hover:text-brand-100"
                  href="https://www.cognitolabs.eu"
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  CognitioLabs.eu
                </a>
                <span className="text-slate-600">|</span>
                <a
                  className="rounded-full border border-brand-500/40 px-3 py-1 text-brand-200 transition hover:border-brand-400 hover:text-brand-100"
                  href="https://github.com/cognitolabs-ai"
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  GitHub @cognitolabs-ai
                </a>
              </div>
            </div>
            <p className="max-w-3xl text-sm text-slate-400">
              Configure base models, datasets, RLHF flavours, and performance instrumentation directly in your browser.
              Download a ready-to-run notebook bundle aligned with your selections.
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 pb-16 pt-10">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <form className="card-surface space-y-10 p-8" onSubmit={handleGenerate}>
            <FormSection
              title="Model & Strategy"
              info="Select the base model, provider, accelerator, and fine-tuning approach."
            >
              <div className="fieldset-grid">
                <Field label="Provider" info="Load a model from the Hugging Face hub or a local checkpoint." htmlFor="provider">
                  <select
                    id="provider"
                    className="input-base"
                    value={config.provider}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        provider: event.target.value as UiConfig['provider']
                      }))
                    }
                  >
                    <option value="hf_hub">hf_hub</option>
                    <option value="hf_local">hf_local</option>
                  </select>
                </Field>
                <Field
                  label="Model Identifier"
                  info="Full identifier of the base model, e.g. mistralai/Mistral-7B-Instruct-v0.3."
                  htmlFor="model_id"
                >
                  <input
                    id="model_id"
                    className="input-base"
                    value={config.model_id}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        model_id: event.target.value
                      }))
                    }
                    required
                  />
                </Field>
                <Field
                  label="Seed"
                  info="Set a deterministic random seed (leave empty to allow randomness)."
                  htmlFor="seed"
                >
                  <input
                    id="seed"
                    className="input-base"
                    type="number"
                    min={0}
                    step={1}
                    value={config.seed?.toString() ?? ''}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        seed: event.target.value === '' ? undefined : Number(event.target.value)
                      }))
                    }
                  />
                </Field>
              </div>

              <div className="flex flex-wrap items-center gap-4">
                <fieldset className="space-y-2">
                  <legend className="text-xs uppercase tracking-wide text-slate-400">Tuning Strategy</legend>
                  <div className="flex flex-wrap gap-2">
                    {(['full', 'lora', 'qlora'] as UiConfig['tune_type'][]).map((option) => (
                      <button
                        key={option}
                        type="button"
                        className={`toggle-pill ${config.tune_type === option ? 'active' : ''}`}
                        onClick={() => handleTuneTypeChange(option)}
                      >
                        {option.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </fieldset>

                <fieldset className="space-y-2">
                  <legend className="text-xs uppercase tracking-wide text-slate-400">Accelerator</legend>
                  <div className="flex gap-2">
                    {(['baseline', 'unsloth'] as UiConfig['accelerator'][]).map((option) => (
                      <button
                        key={option}
                        type="button"
                        className={`toggle-pill ${config.accelerator === option ? 'active' : ''}`}
                        onClick={() => handleAcceleratorChange(option)}
                      >
                        {option === 'unsloth' ? 'Unsloth (PEFT)' : 'Baseline'}
                      </button>
                    ))}
                  </div>
                </fieldset>

                <fieldset className="space-y-2">
                  <legend className="text-xs uppercase tracking-wide text-slate-400">Training Mode</legend>
                  <div className="flex flex-wrap gap-2">
                    {(['supervised', 'grpo', 'dpo', 'orpo'] as TrainingMode[]).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={`toggle-pill ${config.training_mode === mode ? 'active' : ''}`}
                        onClick={() => handleTrainingModeChange(mode)}
                      >
                        {mode.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </fieldset>
              </div>

              {config.accelerator === 'unsloth' && (
                <div className="rounded-xl border border-brand-500/40 bg-brand-500/10 p-4 text-sm text-brand-100">
                  Unsloth accelerator selected. The generator will emit Unsloth-specific imports and FastLanguageModel
                  helpers for LoRA/QLoRA runs. Full fine-tuning automatically falls back to baseline.
                </div>
              )}
            </FormSection>
            {config.tune_type === 'lora' && config.lora && (
              <FormSection title="LoRA Settings" info="Configure rank, alpha, dropout, and module targeting.">
                <div className="fieldset-grid">
                  <NumberField
                    id="lora_rank"
                    label="Rank (r)"
                    info="Dimensionality of the LoRA update matrices."
                    value={config.lora.r.toString()}
                    min="1"
                    step="1"
                    onChange={(value) =>
                      setConfig((prev) => ({
                        ...prev,
                        lora: {
                          ...(prev.lora ?? clone(DEFAULT_LORA)),
                          r: parseOrDefault(value, DEFAULT_LORA.r)
                        }
                      }))
                    }
                  />
                  <NumberField
                    id="lora_alpha"
                    label="Alpha"
                    info="Scaling applied to LoRA updates."
                    value={config.lora.alpha.toString()}
                    step="1"
                    min="0"
                    onChange={(value) =>
                      setConfig((prev) => ({
                        ...prev,
                        lora: {
                          ...(prev.lora ?? clone(DEFAULT_LORA)),
                          alpha: parseOrDefault(value, DEFAULT_LORA.alpha)
                        }
                      }))
                    }
                  />
                  <NumberField
                    id="lora_dropout"
                    label="Dropout"
                    info="Dropout applied to LoRA adapters."
                    value={config.lora.dropout.toString()}
                    step="0.01"
                    min="0"
                    max="1"
                    onChange={(value) =>
                      setConfig((prev) => ({
                        ...prev,
                        lora: {
                          ...(prev.lora ?? clone(DEFAULT_LORA)),
                          dropout: parseOrDefault(value, DEFAULT_LORA.dropout)
                        }
                      }))
                    }
                  />
                </div>
                <div className="fieldset-grid">
                  <Field
                    label="Target modules"
                    info="Comma separated module names to receive LoRA adapters."
                    htmlFor="lora_modules"
                  >
                    <input
                      id="lora_modules"
                      className="input-base"
                      placeholder="q_proj, v_proj"
                      value={(config.lora.target_modules ?? []).join(', ')}
                      onChange={(event) => {
                        const modules = event.target.value
                          .split(',')
                          .map((item) => item.trim())
                          .filter(Boolean);
                        setConfig((prev) => ({
                          ...prev,
                          lora: {
                            ...(prev.lora ?? clone(DEFAULT_LORA)),
                            target_modules: modules.length > 0 ? modules : undefined
                          }
                        }));
                      }}
                    />
                  </Field>
                  <Field label="Bias" info="Bias handling for LoRA adapters." htmlFor="lora_bias">
                    <select
                      id="lora_bias"
                      className="input-base"
                      value={config.lora.bias ?? 'none'}
                      onChange={(event) =>
                        setConfig((prev) => ({
                          ...prev,
                          lora: {
                            ...(prev.lora ?? clone(DEFAULT_LORA)),
                            bias: event.target.value as NonNullable<UiConfig['lora']>['bias']
                          }
                        }))
                      }
                    >
                      <option value="none">none</option>
                      <option value="all">all</option>
                      <option value="lora_only">lora_only</option>
                    </select>
                  </Field>
                </div>
              </FormSection>
            )}

            {config.tune_type === 'qlora' && (
              <FormSection title="QLoRA Settings" info="Configure 4-bit quantization parameters for QLoRA.">
                <div className="fieldset-grid">
                  <Field
                    label="Quant type"
                    info="BitsAndBytes 4-bit quantisation flavour."
                    htmlFor="qlora_quant"
                  >
                    <select
                      id="qlora_quant"
                      className="input-base"
                      value={config.qlora?.bnb_4bit_quant_type ?? DEFAULT_QLORA.bnb_4bit_quant_type}
                      onChange={(event) =>
                        setConfig((prev) => ({
                          ...prev,
                          qlora: {
                            ...(prev.qlora ?? clone(DEFAULT_QLORA)),
                            bnb_4bit_quant_type: event.target.value as NonNullable<UiConfig['qlora']>['bnb_4bit_quant_type']
                          }
                        }))
                      }
                    >
                      <option value="nf4">nf4</option>
                      <option value="fp4">fp4</option>
                    </select>
                  </Field>
                  <Field
                    label="Compute dtype"
                    info="Precision used during 4-bit training."
                    htmlFor="qlora_dtype"
                  >
                    <select
                      id="qlora_dtype"
                      className="input-base"
                      value={config.qlora?.bnb_4bit_compute_dtype ?? DEFAULT_QLORA.bnb_4bit_compute_dtype}
                      onChange={(event) =>
                        setConfig((prev) => ({
                          ...prev,
                          qlora: {
                            ...(prev.qlora ?? clone(DEFAULT_QLORA)),
                            bnb_4bit_compute_dtype: event.target.value as NonNullable<UiConfig['qlora']>['bnb_4bit_compute_dtype']
                          }
                        }))
                      }
                    >
                      <option value="bfloat16">bfloat16</option>
                      <option value="float16">float16</option>
                      <option value="float32">float32</option>
                    </select>
                  </Field>
                  <Field
                    label="Double quantisation"
                    info="Enable nested quantisation (double quant) for VRAM savings."
                    htmlFor="qlora_double"
                  >
                    <select
                      id="qlora_double"
                      className="input-base"
                      value={String(config.qlora?.bnb_4bit_use_double_quant ?? DEFAULT_QLORA.bnb_4bit_use_double_quant)}
                      onChange={(event) =>
                        setConfig((prev) => ({
                          ...prev,
                          qlora: {
                            ...(prev.qlora ?? clone(DEFAULT_QLORA)),
                            bnb_4bit_use_double_quant: event.target.value === 'true'
                          }
                        }))
                      }
                    >
                      <option value="true">true</option>
                      <option value="false">false</option>
                    </select>
                  </Field>
                </div>
              </FormSection>
            )}
            <FormSection title="Dataset" info="Configure data source, format, and split ratios.">
              <div className="fieldset-grid">
                <Field label="Source" info="Load a dataset from Hugging Face or reference a local file." htmlFor="dataset_source">
                  <select
                    id="dataset_source"
                    className="input-base"
                    value={config.dataset.source}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        dataset: {
                          ...prev.dataset,
                          source: event.target.value as UiConfig['dataset']['source'],
                          id: event.target.value === 'hf_dataset_id' ? prev.dataset.id ?? 'samsum' : undefined,
                          path:
                            event.target.value === 'upload_local_path'
                              ? prev.dataset.path ?? './data/train.jsonl'
                              : undefined
                        }
                      }))
                    }
                  >
                    <option value="hf_dataset_id">hf_dataset_id</option>
                    <option value="upload_local_path">upload_local_path</option>
                  </select>
                </Field>

                {config.dataset.source === 'hf_dataset_id' ? (
                  <Field
                    label="Dataset ID"
                    info="Identifier of the dataset on Hugging Face (e.g. samsum)."
                    htmlFor="dataset_id"
                  >
                    <input
                      id="dataset_id"
                      className="input-base"
                      value={config.dataset.id ?? ''}
                      onChange={(event) =>
                        setConfig((prev) => ({
                          ...prev,
                          dataset: {
                            ...prev.dataset,
                            id: event.target.value
                          }
                        }))
                      }
                      required
                    />
                  </Field>
                ) : (
                  <Field
                    label="Dataset path"
                    info="Relative path to the local dataset file (jsonl or csv)."
                    htmlFor="dataset_path"
                  >
                    <input
                      id="dataset_path"
                      className="input-base"
                      value={config.dataset.path ?? ''}
                      onChange={(event) =>
                        setConfig((prev) => ({
                          ...prev,
                          dataset: {
                            ...prev.dataset,
                            path: event.target.value
                          }
                        }))
                      }
                      required
                    />
                  </Field>
                )}

                <Field
                  label="Format"
                  info="Dataset structure to guide preprocessing."
                  htmlFor="dataset_format"
                >
                  <select
                    id="dataset_format"
                    className="input-base"
                    value={config.dataset.format}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        dataset: {
                          ...prev.dataset,
                          format: event.target.value as UiConfig['dataset']['format']
                        }
                      }))
                    }
                  >
                    <option value="jsonl_instr">jsonl_instr</option>
                    <option value="jsonl_chat">jsonl_chat</option>
                    <option value="csv_classification">csv_classification</option>
                  </select>
                </Field>
              </div>

              <div className="fieldset-grid">
                {(['train', 'val', 'test'] as const).map((splitKey) => (
                  <NumberField
                    key={splitKey}
                    id={`split_${splitKey}`}
                    label={`${splitKey.toUpperCase()} split`}
                    info={splitInfo[splitKey]}
                    value={config.dataset.split?.[splitKey]?.toString() ?? ''}
                    min="0"
                    max="1"
                    step="0.01"
                    onChange={(value) => handleDatasetSplit(splitKey, value)}
                  />
                ))}
              </div>
            </FormSection>

            <FormSection title="Hyperparameters" info="Core training hyperparameters for the Trainer.">
              <div className="fieldset-grid">
                <NumberField
                  id="learning_rate"
                  label="Learning rate"
                  info="Base learning rate for the optimiser."
                  value={config.hyperparams.learning_rate.toString()}
                  step="0.0001"
                  min="0"
                  onChange={(value) => updateHyperparam(setConfig, 'learning_rate', value)}
                />
                <NumberField
                  id="batch_size_train"
                  label="Train batch"
                  info="Per-device batch size for training."
                  value={config.hyperparams.batch_size_train.toString()}
                  step="1"
                  min="1"
                  onChange={(value) => updateHyperparam(setConfig, 'batch_size_train', value)}
                />
                <NumberField
                  id="batch_size_eval"
                  label="Eval batch"
                  info="Per-device batch size for evaluation."
                  value={config.hyperparams.batch_size_eval?.toString() ?? ''}
                  step="1"
                  min="1"
                  onChange={(value) => updateHyperparam(setConfig, 'batch_size_eval', value, true)}
                />
                <NumberField
                  id="num_epochs"
                  label="Epochs"
                  info="Number of epochs to train."
                  value={config.hyperparams.num_epochs.toString()}
                  step="1"
                  min="1"
                  onChange={(value) => updateHyperparam(setConfig, 'num_epochs', value)}
                />
                <NumberField
                  id="gradient_accumulation"
                  label="Grad accumulation"
                  info="Gradient accumulation steps."
                  value={config.hyperparams.gradient_accumulation.toString()}
                  step="1"
                  min="1"
                  onChange={(value) => updateHyperparam(setConfig, 'gradient_accumulation', value)}
                />
                <NumberField
                  id="weight_decay"
                  label="Weight decay"
                  info="L2 weight decay (optional)."
                  value={config.hyperparams.weight_decay?.toString() ?? ''}
                  step="0.001"
                  min="0"
                  onChange={(value) => updateHyperparam(setConfig, 'weight_decay', value, true)}
                />
                <Field
                  label="Scheduler"
                  info="Learning rate scheduler strategy."
                  htmlFor="lr_scheduler"
                >
                  <input
                    id="lr_scheduler"
                    className="input-base"
                    value={config.hyperparams.lr_scheduler ?? ''}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        hyperparams: {
                          ...prev.hyperparams,
                          lr_scheduler: event.target.value || undefined
                        }
                      }))
                    }
                  />
                </Field>
                <NumberField
                  id="warmup_ratio"
                  label="Warmup ratio"
                  info="Fraction of steps used for learning rate warmup."
                  value={config.hyperparams.warmup_ratio.toString()}
                  step="0.01"
                  min="0"
                  max="1"
                  onChange={(value) => updateHyperparam(setConfig, 'warmup_ratio', value)}
                />
                <NumberField
                  id="max_seq_len"
                  label="Max seq length"
                  info="Maximum sequence length after tokenization."
                  value={config.hyperparams.max_seq_len.toString()}
                  step="64"
                  min="1"
                  onChange={(value) => updateHyperparam(setConfig, 'max_seq_len', value)}
                />
              </div>
            </FormSection>
            {config.training_mode !== 'supervised' && rlValues && (
              <FormSection
                title="Reinforcement Learning Settings"
                info="Configure reward models, preference datasets, and rollout parameters."
              >
                <div className="fieldset-grid">
                  {config.training_mode === 'grpo' && (
                    <>
                      <Field
                        label="Reward model"
                        info="Model used to score completions during GRPO."
                        htmlFor="rl_reward_model"
                      >
                        <input
                          id="rl_reward_model"
                          className="input-base"
                          value={rlValues.reward_model ?? ''}
                          onChange={(event) =>
                            updateRl((current) => ({
                              ...current,
                              reward_model: event.target.value
                            }))
                          }
                          required
                        />
                      </Field>
                      <Field
                        label="Rollout prompts path"
                        info="JSONL file with prompts used to generate rollouts."
                        htmlFor="rl_rollout_prompts"
                      >
                        <input
                          id="rl_rollout_prompts"
                          className="input-base"
                          value={rlValues.rollout_prompts_path ?? ''}
                          onChange={(event) =>
                            updateRl((current) => ({
                              ...current,
                              rollout_prompts_path: event.target.value
                            }))
                          }
                          required
                        />
                      </Field>
                      <NumberField
                        id="rl_generations"
                        label="Generations per prompt"
                        info="Number of completions sampled per prompt."
                        value={rlValues.num_generations?.toString() ?? ''}
                        step="1"
                        min="1"
                        onChange={(value) =>
                          updateRl((current) => ({
                            ...current,
                            num_generations: value === '' ? undefined : Number(value)
                          }))
                        }
                      />
                      <NumberField
                        id="rl_target_kl"
                        label="Target KL"
                        info="Target KL divergence for policy regularisation."
                        value={rlValues.target_kl?.toString() ?? ''}
                        step="0.5"
                        min="0"
                        onChange={(value) =>
                          updateRl((current) => ({
                            ...current,
                            target_kl: value === '' ? undefined : Number(value)
                          }))
                        }
                      />
                    </>
                  )}

                  {(config.training_mode === 'dpo' || config.training_mode === 'orpo') && (
                    <>
                      <Field
                        label="Preference dataset"
                        info="Dataset containing chosen/rejected pairs for DPO/ORPO."
                        htmlFor="rl_preference_dataset"
                      >
                        <input
                          id="rl_preference_dataset"
                          className="input-base"
                          value={rlValues.preference_dataset ?? ''}
                          onChange={(event) =>
                            updateRl((current) => ({
                              ...current,
                              preference_dataset: event.target.value
                            }))
                          }
                          required
                        />
                      </Field>
                      <Field
                        label="Reference model"
                        info="Reference policy used for divergence penalties."
                        htmlFor="rl_reference_model"
                      >
                        <input
                          id="rl_reference_model"
                          className="input-base"
                          value={rlValues.reference_model ?? ''}
                          onChange={(event) =>
                            updateRl((current) => ({
                              ...current,
                              reference_model: event.target.value
                            }))
                          }
                          required
                        />
                      </Field>
                      <NumberField
                        id="rl_beta"
                        label="Beta"
                        info="Inverse temperature for DPO/ORPO objectives."
                        value={rlValues.beta?.toString() ?? ''}
                        step="0.05"
                        min="0"
                        onChange={(value) =>
                          updateRl((current) => ({
                            ...current,
                            beta: value === '' ? undefined : Number(value)
                          }))
                        }
                      />
                      <NumberField
                        id="rl_warmup"
                        label="Warmup steps"
                        info="Optional warmup steps before enabling RL-specific losses."
                        value={rlValues.warmup_steps?.toString() ?? ''}
                        step="10"
                        min="0"
                        onChange={(value) =>
                          updateRl((current) => ({
                            ...current,
                            warmup_steps: value === '' ? undefined : Number(value)
                          }))
                        }
                      />
                    </>
                  )}

                  <NumberField
                    id="rl_max_steps"
                    label="Max steps"
                    info="Cap the number of RL steps or epochs."
                    value={rlValues.max_steps?.toString() ?? ''}
                    step="10"
                    min="0"
                    onChange={(value) =>
                      updateRl((current) => ({
                        ...current,
                        max_steps: value === '' ? undefined : Number(value)
                      }))
                    }
                  />
                </div>
              </FormSection>
            )}
            <FormSection
              title="HRM Overlay (Experimental)"
              info="Add a tiny recursive controller with adaptive computation time (ACT) on top of your tuning method."
            >
              <div className="flex flex-wrap items-center gap-3">
                <label className="inline-flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={isHrmEnabled}
                    onChange={(event) => handleHrmToggle(event.target.checked)}
                  />
                  Enable HRM overlay
                </label>
                {isHrmEnabled && (
                  <div className="flex flex-wrap gap-2">
                    {HRM_PRESET_OPTIONS.map((preset) => (
                      <button
                        key={preset.key}
                        type="button"
                        className="toggle-pill"
                        onClick={() => applyHrmPreset(preset.key)}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {isHrmEnabled && (
                <div className="space-y-6">
                  <div className="fieldset-grid">
                    <NumberField
                      id="hrm_max_steps"
                      label="Max steps"
                      info="Upper bound for recursive steps per token."
                      value={hrmValues.max_steps.toString()}
                      min="1"
                      max="16"
                    step="1"
                    onChange={(value) =>
                      updateHrm((current) => ({
                        ...current,
                        max_steps: Math.min(
                          Math.max(Math.round(parseOrDefault(value, current.max_steps)), 1),
                          16
                        )
                      }))
                    }
                  />
                    <NumberField
                      id="hrm_threshold"
                      label="ACT threshold"
                      info="Halting threshold for ACT controller."
                      value={(hrmValues.act?.threshold ?? 0.6).toString()}
                      min="0.3"
                      max="0.9"
                      step="0.05"
                      onChange={(value) =>
                        updateHrm((current) => ({
                          ...current,
                          act: {
                            ...(current.act ?? { enabled: true, threshold: 0.6, penalty: 0.02 }),
                            threshold: parseOrDefault(value, current.act?.threshold ?? 0.6)
                          }
                        }))
                      }
                    />
                    <NumberField
                      id="hrm_penalty"
                      label="ACT penalty"
                      info="Penalty applied when exceeding ACT budget."
                      value={(hrmValues.act?.penalty ?? 0.02).toString()}
                      min="0"
                      max="0.05"
                      step="0.005"
                      onChange={(value) =>
                        updateHrm((current) => ({
                          ...current,
                          act: {
                            ...(current.act ?? { enabled: true, threshold: 0.6, penalty: 0.02 }),
                            penalty: parseOrDefault(value, current.act?.penalty ?? 0.02)
                          }
                        }))
                      }
                    />
                  </div>

                  <div className="fieldset-grid">
                    <Field label="Controller type" info="Controller architecture used for HRM." htmlFor="hrm_controller_type">
                      <select
                        id="hrm_controller_type"
                        className="input-base"
                        value={hrmValues.controller?.type ?? 'tiny_mlp'}
                        onChange={(event) =>
                          updateHrm((current) => ({
                            ...current,
                            controller: {
                              ...(current.controller ?? { type: 'tiny_mlp', hidden: 128, layers: 2, dropout: 0.1 }),
                              type: event.target.value as NonNullable<HrmConfig['controller']>['type']
                            }
                          }))
                        }
                      >
                        <option value="tiny_mlp">tiny_mlp</option>
                        <option value="gru">gru</option>
                        <option value="tiny_transformer">tiny_transformer</option>
                      </select>
                    </Field>
                    <NumberField
                      id="hrm_hidden"
                      label="Controller hidden size"
                      info="Hidden dimension for the controller network."
                      value={(hrmValues.controller?.hidden ?? 128).toString()}
                      min="32"
                      max="256"
                      step="16"
                      onChange={(value) =>
                        updateHrm((current) => ({
                          ...current,
                          controller: {
                            ...(current.controller ?? { type: 'tiny_mlp', hidden: 128, layers: 2, dropout: 0.1 }),
                            hidden: parseOrDefault(value, current.controller?.hidden ?? 128)
                          }
                        }))
                      }
                    />
                    <NumberField
                      id="hrm_layers"
                      label="Controller layers"
                      info="Number of layers in the HRM controller."
                      value={(hrmValues.controller?.layers ?? 2).toString()}
                      min="1"
                      max="3"
                      step="1"
                      onChange={(value) =>
                        updateHrm((current) => ({
                          ...current,
                          controller: {
                            ...(current.controller ?? { type: 'tiny_mlp', hidden: 128, layers: 2, dropout: 0.1 }),
                            layers: parseOrDefault(value, current.controller?.layers ?? 2)
                          }
                        }))
                      }
                    />
                    <NumberField
                      id="hrm_dropout"
                      label="Controller dropout"
                      info="Dropout applied within the controller."
                      value={(hrmValues.controller?.dropout ?? 0.1).toString()}
                      min="0"
                      max="0.3"
                      step="0.01"
                      onChange={(value) =>
                        updateHrm((current) => ({
                          ...current,
                          controller: {
                            ...(current.controller ?? { type: 'tiny_mlp', hidden: 128, layers: 2, dropout: 0.1 }),
                            dropout: parseOrDefault(value, current.controller?.dropout ?? 0.1)
                          }
                        }))
                      }
                    />
                  </div>

                  <div className="flex flex-wrap items-center gap-4">
                    <label className="inline-flex items-center gap-2 text-sm text-slate-300">
                      <input
                        type="checkbox"
                        checked={hrmValues.eval?.compute_vs_quality_curve ?? true}
                        onChange={(event) =>
                          updateHrm((current) => ({
                            ...current,
                            eval: {
                              ...(current.eval ?? { compute_vs_quality_curve: true, budgets: [1, 2, 4, 8] }),
                              compute_vs_quality_curve: event.target.checked
                            }
                          }))
                        }
                      />
                      Log compute vs. quality curve
                    </label>
                    <Field
                      label="Budgets"
                      info="Comma-separated budgets to evaluate (e.g., 1,2,4,8)."
                      htmlFor="hrm_budgets"
                    >
                      <input
                        id="hrm_budgets"
                        className="input-base"
                        value={(hrmValues.eval?.budgets ?? [1, 2, 4, 8]).join(', ')}
                        onChange={(event) =>
                          updateHrm((current) => ({
                            ...current,
                            eval: {
                              ...(current.eval ?? { compute_vs_quality_curve: true, budgets: [1, 2, 4, 8] }),
                              budgets: event.target.value
                                .split(',')
                                .map((item) => Number(item.trim()))
                                .filter((value) => !Number.isNaN(value) && value >= 1 && value <= 16)
                            }
                          }))
                        }
                      />
                    </Field>
                  </div>
                </div>
              )}
            </FormSection>
            <FormSection title="Logging, Device & Outputs" info="Control logging destinations, device placement, and checkpoints.">
              <div className="fieldset-grid">
                <Field label="Logging" info="Choose between TensorBoard logging or disable logging." htmlFor="logs">
                  <select
                    id="logs"
                    className="input-base"
                    value={config.logs}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        logs: event.target.value as UiConfig['logs']
                      }))
                    }
                  >
                    <option value="tensorboard">tensorboard</option>
                    <option value="none">none</option>
                  </select>
                </Field>
                <Field label="Device" info="Preferred accelerator for the run." htmlFor="device">
                  <select
                    id="device"
                    className="input-base"
                    value={config.hw.device}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        hw: {
                          ...prev.hw,
                          device: event.target.value as UiConfig['hw']['device']
                        }
                      }))
                    }
                  >
                    <option value="auto">auto</option>
                    <option value="cuda">cuda</option>
                    <option value="cpu">cpu</option>
                  </select>
                </Field>
                <Field label="Mixed precision" info="Precision mode for training." htmlFor="precision">
                  <select
                    id="precision"
                    className="input-base"
                    value={config.hw.mixed_precision}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        hw: {
                          ...prev.hw,
                          mixed_precision: event.target.value as UiConfig['hw']['mixed_precision']
                        }
                      }))
                    }
                  >
                    <option value="bf16">bf16</option>
                    <option value="fp16">fp16</option>
                    <option value="none">none</option>
                  </select>
                </Field>
                <Field
                  label="Output directory"
                  info="Directory where checkpoints and artefacts will be stored."
                  htmlFor="output_dir"
                >
                  <input
                    id="output_dir"
                    className="input-base"
                    value={config.artifacts.output_dir}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        artifacts: {
                          ...prev.artifacts,
                          output_dir: event.target.value
                        }
                      }))
                    }
                  />
                </Field>
                <Field
                  label="Save strategy"
                  info="Checkpoint saving cadence (steps, epoch, etc.)."
                  htmlFor="save_strategy"
                >
                  <input
                    id="save_strategy"
                    className="input-base"
                    value={config.artifacts.save_strategy ?? ''}
                    onChange={(event) =>
                      setConfig((prev) => ({
                        ...prev,
                        artifacts: {
                          ...prev.artifacts,
                          save_strategy: event.target.value || undefined
                        }
                      }))
                    }
                  />
                </Field>
                <NumberField
                  id="save_total_limit"
                  label="Save total limit"
                  info="Maximum number of checkpoints kept on disk."
                  value={(config.artifacts.save_total_limit ?? 2).toString()}
                  step="1"
                  min="1"
                  onChange={(value) =>
                    setConfig((prev) => ({
                      ...prev,
                      artifacts: {
                        ...prev.artifacts,
                        save_total_limit: parseOrDefault(value, prev.artifacts.save_total_limit ?? 2)
                      }
                    }))
                  }
                />
              </div>
            </FormSection>

            <FormSection
              title="torch.compile & Profiler"
              info="Optimise execution with torch.compile and capture traces with torch.profiler."
            >
              <div className="grid gap-6 md:grid-cols-2">
                <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-slate-100">torch.compile()</h4>
                    <label className="inline-flex items-center gap-2 text-xs text-slate-300">
                      <span>Enable</span>
                      <input
                        type="checkbox"
                        checked={compileState.enabled}
                        onChange={(event) =>
                          setConfig((prev) => ({
                            ...prev,
                            compile: {
                              ...(prev.compile ?? { ...DEFAULT_COMPILE }),
                              enabled: event.target.checked
                            }
                          }))
                        }
                      />
                    </label>
                  </div>
                  <div className="mt-4 space-y-3">
                    <Field label="Backend" info="Compilation backend passed to torch.compile." htmlFor="compile_backend">
                      <select
                        id="compile_backend"
                        className="input-base"
                        value={compileState.backend}
                        onChange={(event) =>
                          setConfig((prev) => ({
                            ...prev,
                            compile: {
                              ...(prev.compile ?? { ...DEFAULT_COMPILE }),
                              backend: event.target.value as NonNullable<UiConfig['compile']>['backend']
                            }
                          }))
                        }
                        disabled={!compileState.enabled}
                      >
                        <option value="inductor">inductor</option>
                        <option value="onnxrt">onnxrt</option>
                        <option value="cudagraphs">cudagraphs</option>
                        <option value="eager">eager</option>
                      </select>
                    </Field>
                    <Field label="Mode" info="Compilation heuristics mode." htmlFor="compile_mode">
                      <select
                        id="compile_mode"
                        className="input-base"
                        value={compileState.mode}
                        onChange={(event) =>
                          setConfig((prev) => ({
                            ...prev,
                            compile: {
                              ...(prev.compile ?? { ...DEFAULT_COMPILE }),
                              mode: event.target.value as NonNullable<UiConfig['compile']>['mode']
                            }
                          }))
                        }
                        disabled={!compileState.enabled}
                      >
                        <option value="default">default</option>
                        <option value="reduce-overhead">reduce-overhead</option>
                        <option value="max-autotune">max-autotune</option>
                        <option value="max-autotune-no-cudagraphs">max-autotune-no-cudagraphs</option>
                      </select>
                    </Field>
                    <label className="flex items-center gap-2 text-xs text-slate-300">
                      <input
                        type="checkbox"
                        checked={compileState.fullgraph ?? false}
                        onChange={(event) =>
                          setConfig((prev) => ({
                            ...prev,
                            compile: {
                              ...(prev.compile ?? { ...DEFAULT_COMPILE }),
                              fullgraph: event.target.checked
                            }
                          }))
                        }
                        disabled={!compileState.enabled}
                      />
                      Fullgraph capture
                    </label>
                    <label className="flex items-center gap-2 text-xs text-slate-300">
                      <input
                        type="checkbox"
                        checked={compileState.dynamic ?? true}
                        onChange={(event) =>
                          setConfig((prev) => ({
                            ...prev,
                            compile: {
                              ...(prev.compile ?? { ...DEFAULT_COMPILE }),
                              dynamic: event.target.checked
                            }
                          }))
                        }
                        disabled={!compileState.enabled}
                      />
                      Prefer dynamic shapes
                    </label>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-slate-100">torch.profiler</h4>
                    <label className="inline-flex items-center gap-2 text-xs text-slate-300">
                      <span>Enable</span>
                      <input
                        type="checkbox"
                        checked={profilerState.enabled}
                        onChange={(event) =>
                          setConfig((prev) => ({
                            ...prev,
                            profiler: {
                              ...(prev.profiler ?? { ...DEFAULT_PROFILER }),
                              enabled: event.target.checked
                            }
                          }))
                        }
                      />
                    </label>
                  </div>
                  <div className="mt-4 space-y-3">
                    <Field
                      label="Activities"
                      info="Select which devices to profile."
                      htmlFor="profiler_activities"
                    >
                      <select
                        id="profiler_activities"
                        multiple
                        className="input-base h-24"
                        value={profilerState.activities ?? ['cpu']}
                        onChange={(event) => {
                          const selected = Array.from(event.target.selectedOptions).map((option) => option.value as 'cpu' | 'cuda' | 'xpu');
                          setConfig((prev) => ({
                            ...prev,
                            profiler: {
                              ...(prev.profiler ?? { ...DEFAULT_PROFILER }),
                              activities: selected.length > 0 ? selected : ['cpu']
                            }
                          }));
                        }}
                        disabled={!profilerState.enabled}
                      >
                        <option value="cpu">cpu</option>
                        <option value="cuda">cuda</option>
                        <option value="xpu">xpu</option>
                      </select>
                    </Field>
                    <div className="grid gap-2 text-xs text-slate-300">
                      {[
                        { key: 'record_shapes', label: 'Record shapes' },
                        { key: 'profile_memory', label: 'Profile memory' },
                        { key: 'with_stack', label: 'Capture stack traces' },
                        { key: 'with_flops', label: 'Estimate FLOPs' }
                      ].map((flag) => (
                        <label key={flag.key} className="inline-flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={Boolean((profilerState as Record<string, unknown>)[flag.key])}
                            onChange={(event) =>
                              setConfig((prev) => ({
                                ...prev,
                                profiler: {
                                  ...(prev.profiler ?? { ...DEFAULT_PROFILER }),
                                  [flag.key]: event.target.checked
                                }
                              }))
                            }
                            disabled={!profilerState.enabled}
                          />
                          {flag.label}
                        </label>
                      ))}
                    </div>
                    <div className="grid gap-3 sm:grid-cols-3">
                      {[
                        { key: 'schedule_wait', label: 'wait', info: 'Profiler wait iterations' },
                        { key: 'schedule_warmup', label: 'warmup', info: 'Profiler warmup iterations' },
                        { key: 'schedule_active', label: 'active', info: 'Profiler active iterations' }
                      ].map((item) => (
                        <NumberField
                          key={item.key}
                          id={`profiler_${item.key}`}
                          label={`Schedule ${item.label}`}
                          info={item.info}
                          value={(profilerState as Record<string, unknown>)[item.key]?.toString() ?? ''}
                          min="0"
                          step="1"
                          onChange={(value) =>
                            setConfig((prev) => ({
                              ...prev,
                              profiler: {
                                ...(prev.profiler ?? { ...DEFAULT_PROFILER }),
                                [item.key]: value === '' ? undefined : Number(value)
                              }
                            }))
                          }
                          disabled={!profilerState.enabled}
                        />
                      ))}
                    </div>
                    <Field
                      label="TensorBoard trace dir"
                      info="Optional directory to export traces for TensorBoard."
                      htmlFor="profiler_trace_dir"
                    >
                      <input
                        id="profiler_trace_dir"
                        className="input-base"
                        value={profilerState.tensorboard_trace_dir ?? ''}
                        onChange={(event) =>
                          setConfig((prev) => ({
                            ...prev,
                            profiler: {
                              ...(prev.profiler ?? { ...DEFAULT_PROFILER }),
                              tensorboard_trace_dir: event.target.value || undefined
                            }
                          }))
                        }
                        disabled={!profilerState.enabled}
                      />
                    </Field>
                  </div>
                </div>
              </div>
            </FormSection>

            <FormSection title="Evaluation metrics" info="Select evaluation metrics to compute after training.">
              <div className="flex flex-wrap gap-3">
                {METRICS.map((metric) => (
                  <label key={metric} className="inline-flex items-center gap-2 rounded-full border border-slate-700 px-3 py-1 text-xs uppercase tracking-wide text-slate-200">
                    <input
                      type="checkbox"
                      checked={config.eval.metrics.includes(metric)}
                      onChange={() => handleMetricToggle(metric)}
                    />
                    {metric}
                  </label>
                ))}
              </div>
            </FormSection>

            <div className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-800 pt-6">
              <div className="text-sm text-slate-400">
                {status === 'idle' && 'Ready to generate bundle.'}
                {status === 'working' && 'Rendering bundle...'}
                {status === 'success' && 'Bundle generated successfully.'}
                {status === 'error' && 'Failed to generate bundle.'}
              </div>
              <div className="flex gap-3">
                <button
                  type="button"
                  className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-300 transition hover:border-slate-500 hover:text-white"
                  onClick={resetForm}
                >
                  Reset
                </button>
                <button
                  type="submit"
                  className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-brand-500/30 transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={status === 'working'}
                >
                  Generate bundle
                </button>
              </div>
            </div>

            {errors.length > 0 && (
              <div className="rounded-xl border border-red-500/50 bg-red-500/10 p-4 text-sm text-red-100">
                <strong className="font-semibold">Validation issues</strong>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  {errors.map((issue) => (
                    <li key={issue}>{issue}</li>
                  ))}
                </ul>
              </div>
            )}
          </form>
          <aside className="card-surface flex flex-col gap-6 p-6">
            <div>
              <h2 className="text-lg font-semibold text-white">Downloads</h2>
              <p className="mt-2 text-sm text-slate-400">
                Generate a new bundle to refresh links. Links remain valid for this session only.
              </p>
              {downloads.length === 0 ? (
                <p className="mt-4 rounded-lg border border-dashed border-slate-700/80 bg-slate-900/50 p-4 text-sm text-slate-500">
                  No artefacts yet. Configure parameters and click generate.
                </p>
              ) : (
                <ul className="mt-4 space-y-3">
                  {downloads.map((item) => (
                    <li key={item.filename} className="flex items-center justify-between rounded-xl border border-slate-800/80 bg-slate-900/60 p-3">
                      <div>
                        <strong className="text-sm text-white">{item.filename}</strong>
                        <div className="text-xs text-slate-400">{item.description}</div>
                        <small className="text-xs text-slate-500">{item.size}</small>
                      </div>
                      <a
                        className="rounded-full border border-brand-500/40 px-3 py-1 text-xs font-semibold text-brand-100 transition hover:border-brand-400 hover:text-brand-50"
                        href={item.url}
                        download={item.filename}
                      >
                        Download
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="rounded-2xl border border-slate-800/80 bg-slate-900/60 p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Config preview</h3>
                <InfoTooltip
                  label="Config Preview"
                  description="Live JSON preview of the configuration that will be embedded into the generated notebook."
                />
              </div>
              {!validationResult.success && (
                <p className="mt-2 text-xs text-amber-300">Preview may be incomplete until validation succeeds.</p>
              )}
              <pre className="mt-3 max-h-[420px] overflow-auto rounded-xl border border-slate-800/80 bg-slate-950/80 p-4 text-xs leading-relaxed text-slate-200">
{previewConfig}
              </pre>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
interface FieldProps {
  label: string;
  info: string;
  htmlFor?: string;
  children: React.ReactNode;
}

function Field({ label, info, htmlFor, children }: FieldProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-sm text-slate-300">
        {htmlFor ? (
          <label htmlFor={htmlFor} className="font-medium text-slate-200">
            {label}
          </label>
        ) : (
          <span className="font-medium text-slate-200">{label}</span>
        )}
        <InfoTooltip label={label} description={info} />
      </div>
      {children}
    </div>
  );
}

interface NumberFieldProps {
  id: string;
  label: string;
  info: string;
  value: string;
  onChange: (value: string) => void;
  step?: string;
  min?: string;
  max?: string;
  disabled?: boolean;
}

function NumberField({ id, label, info, value, step, min, max, onChange, disabled }: NumberFieldProps) {
  return (
    <Field label={label} info={info} htmlFor={id}>
      <input
        id={id}
        type="number"
        className="input-base"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      />
    </Field>
  );
}

interface FormSectionProps {
  title: string;
  info: string;
  children: React.ReactNode;
}

function FormSection({ title, info, children }: FormSectionProps) {
  return (
    <section className="space-y-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <p className="mt-1 text-sm text-slate-400">{info}</p>
        </div>
      </div>
      <div className="space-y-4">{children}</div>
    </section>
  );
}

const createDownloadLink = (filename: string, parts: BlobPart[], mime: string, description: string): DownloadLink => {
  const blob = new Blob(parts, { type: mime });
  const url = URL.createObjectURL(blob);
  return {
    filename,
    url,
    description,
    size: formatBytes(blob.size)
  };
};

type HyperparamKey = keyof UiConfig['hyperparams'];

const updateHyperparam = (
  setConfig: React.Dispatch<React.SetStateAction<UiConfig>>,
  key: HyperparamKey,
  rawValue: string,
  allowEmpty = false
) => {
  setConfig((prev) => {
    const parsed = Number(rawValue);
    const nextValue = rawValue === '' && allowEmpty ? undefined : Number.isNaN(parsed) ? prev.hyperparams[key] : parsed;
    return {
      ...prev,
      hyperparams: {
        ...prev.hyperparams,
        [key]: nextValue
      }
    };
  });
};

export default App;
