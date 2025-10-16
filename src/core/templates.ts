export interface TemplateSet {
  notebook: string;
  readme: string;
  config: string;
}

const lines = (...rows: string[]): string[] => rows.map((row) => `${row}\n`);
const block = (...rows: string[]): string => rows.join('\n');

const notebookTemplateObject = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    language_info: {
      name: 'python',
      version: '3.10'
    },
    kernelspec: {
      name: 'python3',
      display_name: 'Python 3'
    },
    codex: {
      generated_at: '{{ generated_at_iso }}',
      provider: '{{ config.provider }}',
      model_id: '{{ config.model_id }}',
      tune_type: '{{ config.tune_type }}'
    }
  },
  cells: [
    {
      cell_type: 'markdown',
      metadata: { tags: ['overview'] },
      source: lines(
        '# Codex Fine-Tuning Notebook',
        '',
        'Generated on {{ generated_at_iso }}',
        '',
        '> This notebook was generated automatically from your configuration. Run each cell in order. Adjust hyperparameters only if you understand their impact.'
      )
    },
    {
      cell_type: 'markdown',
      metadata: { tags: ['summary'] },
      source: lines(
        '## Experiment Summary',
        '',
        '| Setting | Value |',
        '| --- | --- |',
        `| Provider | \`{{ config.provider }}\` |`,
        `| Model ID | \`{{ config.model_id }}\` |`,
        `| Tune Type | \`{{ config.tune_type }}\` |`,
        `| Accelerator | \`{{ config.accelerator | default('baseline') }}\` |`,
        `| Training Mode | \`{{ config.training_mode | default('supervised') }}\` |`,
        `| Dataset Source | \`{{ config.dataset.source }}\` {% if config.dataset.id %}(\`{{ config.dataset.id }}\`){% endif %} {% if config.dataset.path %}(\`{{ config.dataset.path }}\`){% endif %} |`,
        `| Format | \`{{ config.dataset.format }}\` |`,
        `| Eval Metrics | {% if config.eval and config.eval.metrics %}\`{{ config.eval.metrics | join(', ') }}\`{% else %}\`n/a\`{% endif %} |`,
        `| Output Dir | \`{{ config.artifacts.output_dir }}\` |`,
        `| Seed | \`{{ config.seed | default('n/a') }}\` |`,
        `| HRM Overlay | {% if config.hrm and config.hrm.enabled %}\`enabled\` (max_steps={{ config.hrm.max_steps | default(4) }}){% else %}\`disabled\`{% endif %} |`
      )
    },
    {
      cell_type: 'markdown',
      metadata: { tags: ['checklist'] },
      source: lines(
        '### Pre-Run Checklist',
        '',
        '- [ ] Review the configuration below.',
        '- [ ] Ensure GPU is available (if required).',
        '- [ ] Authenticate to Hugging Face if pulling private models or datasets.',
        '- [ ] Update dataset paths if you are running locally.'
      )
    },
    {
      cell_type: 'code',
      metadata: { tags: ['config'] },
      execution_count: null,
      outputs: [],
      source: ['CONFIG = {}']
    },
    {
      cell_type: 'markdown',
      metadata: { tags: ['setup'] },
      source: lines(
        '## 1. Environment Setup',
        'Install required packages. Skip installations that you already have pinned in your environment.'
      )
    },
    {
      cell_type: 'code',
      metadata: { tags: ['setup', 'pip'] },
      execution_count: null,
      outputs: [],
      source: [
        '%pip install --quiet --upgrade accelerate datasets evaluate peft tensorboard transformers trl bitsandbytes sentencepiece safetensors numpy'
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['imports'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          'import json',
          'import math',
          'import os',
          'import random',
          'import shutil',
          'import time',
          '',
          'import datasets',
          'import numpy as np',
          'import torch',
          'from datasets import load_dataset',
          "{% if config.accelerator == 'unsloth' %}from unsloth import FastLanguageModel{% endif %}",
          "{% if config.training_mode == 'grpo' %}from trl import GRPOConfig, GRPOTrainer{% elif config.training_mode == 'dpo' %}from trl import DPOTrainer{% elif config.training_mode == 'orpo' %}from trl import ORPOTrainer{% endif %}",
          'from transformers import (',
          '    AutoModelForCausalLM,',
          '    AutoTokenizer,',
          '    BitsAndBytesConfig,',
          '    Trainer,',
          '    TrainingArguments',
          ')',
          'from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler'
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['utilities'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          'def set_seed(seed: int | None) -> None:',
          '    if seed is None:',
          '        return',
          '    random.seed(seed)',
          '    np.random.seed(seed)',
          '    torch.manual_seed(seed)',
          '    torch.cuda.manual_seed_all(seed)',
          '',
          "set_seed(CONFIG.get('seed'))",
          "print('Seed set to', CONFIG.get('seed'))"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['validation'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "required_keys = ['provider', 'model_id', 'tune_type', 'hyperparams', 'dataset', 'eval', 'logs', 'hw', 'artifacts']",
          'missing = [key for key in required_keys if key not in CONFIG]',
          'if missing:',
          "    raise ValueError(f'Missing required config keys: {missing}')",
          "print('Config validation passed. Provider:', CONFIG['provider'])"
        )
      ]
    },
    {
      cell_type: 'markdown',
      metadata: { tags: ['dataset'] },
      source: lines('## 2. Dataset Loading & Preparation')
    },
    {
      cell_type: 'code',
      metadata: { tags: ['dataset', 'load'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "dataset_cfg = CONFIG['dataset']",
          "if dataset_cfg['source'] == 'hf_dataset_id':",
          "    dataset = load_dataset(dataset_cfg['id'])",
          "elif dataset_cfg['source'] == 'upload_local_path':",
          '    data_files = {}',
          "    path = dataset_cfg.get('path')",
          '    if not path:',
          "        raise ValueError('Local dataset path must be provided when source is upload_local_path')",
          '    ext = os.path.splitext(path)[1]',
          "    if ext in {'.json', '.jsonl'}:",
          "        data_files['train'] = path",
          "    elif ext in {'.csv'}:",
          "        data_files['train'] = path",
          '    else:',
          "        raise ValueError(f'Unsupported local dataset extension: {ext}')",
          "    dataset = load_dataset('json' if ext.startswith('.json') else 'csv', data_files=data_files)",
          'else:',
          "    raise ValueError(f\"Unsupported dataset source: {dataset_cfg['source']}\")",
          '',
          "training_mode = CONFIG.get('training_mode', 'supervised')",
          "rl_cfg = CONFIG.get('rl', {})",
          "preference_dataset = None",
          "if training_mode in {'dpo', 'orpo'}:",
          "    preference_id = rl_cfg.get('preference_dataset')",
          "    if not preference_id:",
          "        raise ValueError('Preference dataset is required for DPO/ORPO modes.')",
          "    preference_dataset = load_dataset(preference_id)",
          '',
          "print('Base dataset:', dataset)",
          "if preference_dataset is not None:",
          "    print('Preference dataset summary:', preference_dataset)"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['dataset', 'split'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "split_cfg = dataset_cfg.get('split') or {}",
          "training_mode = CONFIG.get('training_mode', 'supervised')",
          "if training_mode in {'dpo', 'orpo'} and preference_dataset is not None:",
          "    base = preference_dataset",
          "    train_dataset = base['train']",
          "    eval_dataset = base['validation'] if 'validation' in base else train_dataset.select(range(min(200, len(train_dataset))))",
          "elif {'train', 'val'} <= split_cfg.keys():",
          "    dataset = dataset['train'].train_test_split(test_size=split_cfg['val'])",
          "    train_dataset = dataset['train']",
          "    eval_dataset = dataset['test']",
          "elif 'train' in dataset and 'validation' in dataset:",
          "    train_dataset = dataset['train']",
          "    eval_dataset = dataset['validation']",
          'else:',
          "    train_dataset = dataset['train']",
          "    eval_dataset = dataset['train'].select(range(min(100, len(dataset['train']))))",
          '',
          "print('Train samples:', len(train_dataset))",
          "print('Eval samples:', len(eval_dataset))"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['tokenizer'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "accelerator = CONFIG.get('accelerator', 'baseline')",
          "max_length = CONFIG['hyperparams'].get('max_seq_len', 2048)",
          "{% if config.accelerator == 'unsloth' %}",
          "tokenizer = FastLanguageModel.get_tokenizer(CONFIG['model_id'], max_seq_length=max_length)",
          "{% else %}",
          "tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_id'], use_fast=True)",
          "{% endif %}",
          'if tokenizer.pad_token is None:',
          '    tokenizer.pad_token = tokenizer.eos_token',
          "print('Tokenizer loaded')"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['preprocess'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "max_length = CONFIG['hyperparams'].get('max_seq_len', 1024)",
          "training_mode = CONFIG.get('training_mode', 'supervised')",
          'if training_mode in {"dpo", "orpo", "grpo"}:',
          '    train_tokenized = train_dataset',
          '    eval_tokenized = eval_dataset',
          'else:',
          '    def tokenize_function(example):',
          "        if CONFIG['dataset']['format'] == 'csv_classification':",
          "            text = example['text']",
          "        elif CONFIG['dataset']['format'] == 'jsonl_chat':",
          "            messages = example['messages']",
          "            text = '\\n'.join([f\"{msg['role']}: {msg['content']}\" for msg in messages])",
          '        else:',
          "            text = example.get('instruction', '') + '\\n' + example.get('input', '') + '\\n' + example.get('output', '')",
          '        return tokenizer(text, truncation=True, max_length=max_length)',
          '',
          '    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)',
          '    eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)',
          "print('Tokenization complete for mode:', training_mode)"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['model'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "tune_type = CONFIG['tune_type']",
          "accelerator = CONFIG.get('accelerator', 'baseline')",
          "device_choice = CONFIG['hw']['device']",
          "max_seq_len = CONFIG['hyperparams'].get('max_seq_len', 2048)",
          'model_kwargs = {}',
          'quant_config = None',
          "if accelerator == 'unsloth':",
          "    load_in_4bit = tune_type == 'qlora'",
          "    model, tokenizer = FastLanguageModel.from_pretrained(",
          "        model_name=CONFIG['model_id'],",
          '        max_seq_length=max_seq_len,',
          '        dtype=None,',
          '        load_in_4bit=load_in_4bit',
          '    )',
          "    if tune_type in {'lora', 'qlora'}:",
          "        lora_cfg = CONFIG.get('lora') or CONFIG.get('qlora') or {}",
          '        model = FastLanguageModel.get_peft_model(',
          '            model,',
          "            r=lora_cfg.get('r', 16),",
          "            lora_alpha=lora_cfg.get('alpha', 32),",
          "            lora_dropout=lora_cfg.get('dropout', 0.05),",
          "            target_modules=lora_cfg.get('target_modules'),",
          "            bias=lora_cfg.get('bias', 'none')",
          '        )',
          '    FastLanguageModel.for_training(model)',
          'else:',
          "    if tune_type == 'qlora':",
          "        q_cfg = CONFIG.get('qlora', {})",
          '        quant_config = BitsAndBytesConfig(',
          "            load_in_4bit=True,",
          "            bnb_4bit_quant_type=q_cfg.get('bnb_4bit_quant_type', 'nf4'),",
          "            bnb_4bit_use_double_quant=q_cfg.get('bnb_4bit_use_double_quant', True),",
          "            bnb_4bit_compute_dtype=getattr(torch, q_cfg.get('bnb_4bit_compute_dtype', 'float16'))",
          '        )',
          "        model_kwargs['quantization_config'] = quant_config",
          '    model = AutoModelForCausalLM.from_pretrained(',
          "        CONFIG['model_id'],",
          "        device_map='auto' if device_choice == 'auto' else device_choice,",
          '        **model_kwargs',
          '    )',
          "    if tune_type == 'lora':",
          '        from peft import LoraConfig, get_peft_model',
          "        l_cfg = CONFIG.get('lora', {})",
          '        peft_config = LoraConfig(',
          "            r=l_cfg.get('r', 16),",
          "            lora_alpha=l_cfg.get('alpha', 32),",
          "            lora_dropout=l_cfg.get('dropout', 0.05),",
          "            bias=l_cfg.get('bias', 'none'),",
          "            target_modules=l_cfg.get('target_modules')",
          '        )',
          '        model = get_peft_model(model, peft_config)',
          '        model.print_trainable_parameters()',
          "    if tune_type == 'qlora':",
          "        model.config.use_cache = False",
          "        if hasattr(model, 'gradient_checkpointing_enable'):",
          '            model.gradient_checkpointing_enable()',
          '',
          "compile_cfg = CONFIG.get('compile', {})",
          "if compile_cfg.get('enabled', False):",
          '    model = torch.compile(',
          '        model,',
          "        backend=compile_cfg.get('backend', 'inductor'),",
          "        mode=compile_cfg.get('mode', 'default'),",
          "        fullgraph=compile_cfg.get('fullgraph', False),",
          "        dynamic=compile_cfg.get('dynamic')",
          '    )',
          '',
          "print('Model prepared with accelerator:', accelerator, 'and tune type:', tune_type)"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['training'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "hyper = CONFIG['hyperparams']",
          "training_mode = CONFIG.get('training_mode', 'supervised')",
          "rl_cfg = CONFIG.get('rl', {})",
          'common_args = dict(',
          "    output_dir=CONFIG['artifacts']['output_dir'],",
          "    per_device_train_batch_size=hyper['batch_size_train'],",
          "    per_device_eval_batch_size=hyper.get('batch_size_eval', hyper['batch_size_train']),",
          "    gradient_accumulation_steps=hyper['gradient_accumulation'],",
          "    num_train_epochs=hyper['num_epochs'],",
          "    learning_rate=hyper['learning_rate'],",
          "    weight_decay=hyper.get('weight_decay', 0.0),",
          "    lr_scheduler_type=hyper.get('lr_scheduler', 'cosine'),",
          "    warmup_ratio=hyper['warmup_ratio'],",
          '    logging_steps=10,',
          "    evaluation_strategy='epoch',",
          "    save_strategy=CONFIG['artifacts'].get('save_strategy', 'epoch'),",
          "    save_total_limit=CONFIG['artifacts'].get('save_total_limit', 2),",
          "    report_to=['tensorboard'] if CONFIG['logs'] == 'tensorboard' else []",
          ')',
          'trainer = None',
          'evaluate_callable = None',
          '',
          "if training_mode == 'supervised':",
          '    training_args = TrainingArguments(**common_args)',
          '    trainer = Trainer(',
          '        model=model,',
          '        args=training_args,',
          '        train_dataset=train_tokenized,',
          '        eval_dataset=eval_tokenized,',
          '        tokenizer=tokenizer',
          '    )',
          '    evaluate_callable = trainer.evaluate',
          "elif training_mode == 'grpo':",
          "    if not rl_cfg.get('reward_model'):",
          "        raise ValueError('Reward model is required for GRPO mode.')",
          "    prompts_path = rl_cfg.get('rollout_prompts_path')",
          "    if not prompts_path:",
          "        raise ValueError('Rollout prompts path is required for GRPO mode.')",
          '    prompts: list[str] = []',
          "    with open(prompts_path, 'r', encoding='utf-8') as fp:",
          '        for line in fp:',
          '            payload = line.strip()',
          '            if not payload:',
          '                continue',
          '            try:',
          '                record = json.loads(payload)',
          '            except json.JSONDecodeError:',
          '                record = payload',
          '            if isinstance(record, dict):',
          "                prompt = record.get('prompt') or record.get('input') or record.get('instruction') or record.get('question')",
          '                prompts.append(prompt or json.dumps(record))',
          '            else:',
          '                prompts.append(str(record))',
          '    grpo_config = GRPOConfig(',
          "        num_generations=rl_cfg.get('num_generations', 4),",
          "        target_kl=rl_cfg.get('target_kl', 6),",
          "        max_steps=rl_cfg.get('max_steps', 256),",
          "        beta=rl_cfg.get('beta', 0.05)",
          '    )',
          '    training_args = TrainingArguments(**common_args)',
          '    trainer = GRPOTrainer(',
          '        model=model,',
          "        reward_model=rl_cfg['reward_model'],",
          '        tokenizer=tokenizer,',
          '        args=training_args,',
          '        prompts=prompts,',
          '        grpo_config=grpo_config',
          '    )',
          "elif training_mode in {'dpo', 'orpo'}:",
          "    if not rl_cfg.get('reference_model'):",
          "        raise ValueError('Reference model is required for DPO/ORPO modes.')",
          "    ref_model = AutoModelForCausalLM.from_pretrained(rl_cfg['reference_model'])",
          '    trainer_cls = DPOTrainer if training_mode == \'dpo\' else ORPOTrainer',
          '    training_args = TrainingArguments(**common_args)',
          '    trainer = trainer_cls(',
          '        model=model,',
          '        ref_model=ref_model,',
          '        args=training_args,',
          "        beta=rl_cfg.get('beta', 0.1),",
          '        train_dataset=train_dataset,',
          '        eval_dataset=eval_dataset,',
          '        tokenizer=tokenizer',
          '    )',
          '    evaluate_callable = getattr(trainer, "evaluate", None)',
          'else:',
          "    raise ValueError(f\"Unsupported training mode: {training_mode}\")",
          '',
          "print('Trainer initialised for mode:', training_mode)"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['training', 'run'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          'if trainer is None:',
          "    raise RuntimeError('Trainer is not initialised. Check configuration.')",
          "prof_cfg = CONFIG.get('profiler', {})",
          'metrics = {}',
          '',
          'def _resolve_activities(names):',
          '    available = []',
          '    for name in names:',
          "        if name == 'cpu':",
          '            available.append(ProfilerActivity.CPU)',
          "        elif name == 'cuda' and torch.cuda.is_available():",
          '            available.append(ProfilerActivity.CUDA)',
          "        elif name == 'xpu' and hasattr(ProfilerActivity, 'XPU'):",
          '            available.append(getattr(ProfilerActivity, \'XPU\'))',
          '    return available or [ProfilerActivity.CPU]',
          '',
          'activities = _resolve_activities(prof_cfg.get(\'activities\', [\'cpu\']))',
          'schedule_cfg = schedule(',
          "    wait=prof_cfg.get('schedule_wait', 1),",
          "    warmup=prof_cfg.get('schedule_warmup', 1),",
          "    active=prof_cfg.get('schedule_active', 3)",
          ')',
          "trace_dir = prof_cfg.get('tensorboard_trace_dir')",
          'handler = tensorboard_trace_handler(trace_dir) if trace_dir else None',
          '',
          "if prof_cfg.get('enabled', False):",
          '    with profile(',
          '        activities=activities,',
          '        schedule=schedule_cfg,',
          "        record_shapes=prof_cfg.get('record_shapes', False),",
          "        profile_memory=prof_cfg.get('profile_memory', False),",
          "        with_stack=prof_cfg.get('with_stack', False),",
          "        with_flops=prof_cfg.get('with_flops', False),",
          '        on_trace_ready=handler',
          '    ) as prof:',
          '        train_output = trainer.train()',
          '        prof.step()',
          'else:',
          '    train_output = trainer.train()',
          '',
          'print(train_output)',
          'if callable(evaluate_callable) and training_mode != \'grpo\':',
          '    metrics = evaluate_callable() or {}',
          '    print(metrics)',
          'else:',
          "    print(f'Skipping evaluation for mode: {training_mode}')"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['evaluation'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          'if metrics:',
          "    print('Evaluation metrics:', metrics)",
          'else:',
          "    print('No evaluation metrics captured for this configuration.')"
        )
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['artifacts'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "output_dir = CONFIG['artifacts']['output_dir']",
          'os.makedirs(output_dir, exist_ok=True)',
          'trainer.save_model(output_dir)',
          "if tune_type == 'lora':",
          "    model.save_pretrained(os.path.join(output_dir, 'lora_adapter'))",
          "with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as fp:",
          '    json.dump(CONFIG, fp, indent=2)',
          "print('Artifacts saved to', output_dir)"
        )
      ]
    },
    {
      cell_type: 'markdown',
      metadata: { tags: ['hrm'] },
      source: [
        '{% if config.hrm and config.hrm.enabled %}## HRM Overlay Insights\n\nThe following cells summarise the experimental HRM configuration. Replace the placeholder logic with your full HRM implementation once ready.{% endif %}'
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['hrm'] },
      execution_count: null,
      outputs: [],
      source: [
        '{% if config.hrm and config.hrm.enabled %}',
        "hrm_cfg = CONFIG.get('hrm', {})",
        "act_cfg = hrm_cfg.get('act', {})",
        "controller_cfg = hrm_cfg.get('controller', {})",
        "dtype = torch.bfloat16 if CONFIG['hw'].get('mixed_precision') == 'bf16' and torch.cuda.is_available() else torch.float16",
        "print('HRM overlay enabled:')",
        "print('  max_steps       :', hrm_cfg.get('max_steps', 4))",
        "print('  ACT threshold   :', act_cfg.get('threshold', 0.6))",
        "print('  ACT penalty     :', act_cfg.get('penalty', 0.02))",
        "print('  Controller type :', controller_cfg.get('type', 'tiny_mlp'))",
        "print('  Controller hidden/layers/dropout :', controller_cfg.get('hidden', 128), controller_cfg.get('layers', 2), controller_cfg.get('dropout', 0.1))",
        "print('  Selected dtype  :', dtype)",
        "# TODO: initialize your tiny controller and halting head below. This placeholder only logs the configuration.",
        "{% else %}",
        "print('HRM overlay disabled; skipping HRM configuration.')",
        "{% endif %}"
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['hrm', 'evaluation'] },
      execution_count: null,
      outputs: [],
      source: [
        '{% if config.hrm and config.hrm.enabled %}',
        "hrm_cfg = CONFIG.get('hrm', {})",
        "act_cfg = hrm_cfg.get('act', {})",
        "budgets = hrm_cfg.get('eval', {}).get('budgets', [1, 2, 4, 8])",
        "compute_quality_summary = []",
        "for budget in budgets:",
        "    entry = {",
        "        'budget': int(budget),",
        "        'mean_steps': min(int(budget), hrm_cfg.get('max_steps', 4)),",
        "        'act_threshold': act_cfg.get('threshold', 0.6),",
        "        'notes': 'Placeholder – replace with actual HRM evaluation outputs.'",
        "    }",
        "    if 'metrics' in locals():",
        "        entry['quality'] = metrics",
        "    compute_quality_summary.append(entry)",
        "print('HRM compute vs quality (placeholder summary):')",
        "for row in compute_quality_summary:",
        "    print(row)",
        "{% else %}",
        "pass",
        "{% endif %}"
      ]
    },
    {
      cell_type: 'code',
      metadata: { tags: ['smoke-test'] },
      execution_count: null,
      outputs: [],
      source: [
        block(
          "demo_prompt = 'Hello, how can I help you today?'",
          "inputs = tokenizer(demo_prompt, return_tensors='pt')",
          'inputs = {k: v.to(model.device) for k, v in inputs.items()}',
          'with torch.no_grad():',
          '    outputs = model.generate(**inputs, max_new_tokens=64)',
          "print(tokenizer.decode(outputs[0], skip_special_tokens=True))"
        )
      ]
    },
    {
      cell_type: 'markdown',
      metadata: { tags: ['environment'] },
      source: lines(
        '## Environment Snapshot',
        '',
        '```python',
        'import platform',
        'import torch',
        "print('Python', platform.python_version())",
        "print('PyTorch', torch.__version__)",
        "print('CUDA available:', torch.cuda.is_available())",
        '```'
      )
    }
  ]
};

const readmeTemplate = `# Fine-Tuning Notebook Bundle

Generated: \`{{ generated_at_iso }}\`

## Experiment Overview

- **Provider:** \`{{ config.provider }}\`
- **Model ID:** \`{{ config.model_id }}\`
- **Tune Type:** \`{{ config.tune_type }}\`
- **Accelerator:** \`{{ config.accelerator | default('baseline') }}\`
- **Training Mode:** \`{{ config.training_mode | default('supervised') }}\`
- **Dataset Source:** {% if config.dataset.id %}\`{{ config.dataset.id }}\`{% elif config.dataset.path %}\`{{ config.dataset.path }}\`{% else %}\`{{ config.dataset.source }}\`{% endif %}
- **Dataset Format:** \`{{ config.dataset.format }}\`
- **Eval Metrics:** {% if config.eval and config.eval.metrics %}\`{{ config.eval.metrics | join(', ') }}\`{% else %}\`n/a\`{% endif %}
- **Output Directory:** \`{{ config.artifacts.output_dir }}\`
- **Seed:** \`{{ config.seed | default('n/a') }}\`
- **torch.compile:** {% if config.compile and config.compile.enabled %}\`enabled ({{ config.compile.backend }}/{{ config.compile.mode }})\`{% else %}\`disabled\`{% endif %}
- **Profiler:** {% if config.profiler and config.profiler.enabled %}\`enabled ({{ (config.profiler.activities or ['cpu']) | join(', ') }})\`{% else %}\`disabled\`{% endif %}

## Hyperparameters

\`\`\`yaml
{{ config_yaml }}
\`\`\`

## Included Files

| File | Description |
| --- | --- |
| \`fine_tune.ipynb\` | Ready-to-run notebook with setup, validation, training, evaluation, and smoke test cells. |
| \`config.yaml\` | Stable YAML representation of the configuration. |
| \`README.md\` | This quickstart guide. |

## Run Instructions

1. Launch JupyterLab (or Google Colab) and open \`fine_tune.ipynb\`.
2. Run each cell in sequence. The notebook will:
   - install dependencies,
   - validate configuration,
   - load and preprocess the dataset,
   - initialize the model (Full/LoRA/QLoRA),
   - run training, evaluation, and smoke inference,
   - save artifacts to \`{{ config.artifacts.output_dir }}\`.
3. Inspect the generated outputs (metrics, logs, and saved adapters/models).

## Notes

- For LoRA/QLoRA runs ensure \`bitsandbytes\` and compatible GPU drivers are installed.
- If you change any parameter in the UI, regenerate the bundle to keep README/config in sync.
- Keep this bundle under version control to preserve reproducibility.
## Advanced Features
## Advanced Features
- **Unsloth accelerator:** When enabled the notebook switches to the \`unsloth\` fast-path for LoRA/QLoRA training, reducing VRAM usage and wall-clock time.
- **torch.compile:** Toggle compile-time optimisation and choose the backend/mode directly from the UI; the notebook wraps the model with \`torch.compile(...)\` when enabled.
- **PyTorch profiler:** Capture traces with \`torch.profiler\` using your chosen activities, schedule, and optional TensorBoard export directory.`;

const configTemplate = `# Auto-generated configuration for the Codex notebook generator
{{ config_yaml }}
`;

export const defaultTemplates: TemplateSet = {
  notebook: JSON.stringify(notebookTemplateObject, null, 2),
  readme: readmeTemplate,
  config: configTemplate
};
