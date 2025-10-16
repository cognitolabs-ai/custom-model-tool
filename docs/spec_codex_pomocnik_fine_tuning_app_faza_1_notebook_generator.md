# Specifikacija za **Codex pomočnika** pri razvoju spletne aplikacije za fine‑tuning AI modelov

**Verzija:** 1.0  
**Datum:** 10. oktober 2025  
**Avtor:** (vnesi)  
**Status:** Osnutek za izvedbo Faza 1 (Notebook Generator)

---

## 0. Povzetek

Zgradili bomo spletno aplikacijo, ki vodi uporabnika skozi fine‑tuning LLM/ML modelov. V **Fazi 1** aplikacija generira **Jupyter Notebook**, ki izvede celoten postopek fine‑tuninga na podlagi izbranih parametrov (osnovni model, LoRA/QLoRA, datasets, metrike, strojna oprema …). Implementacija poteka v **korakih/fazah**, vsak korak ima jasne **testne kriterije** in **merila sprejema**. Kasnejše faze dodajo backend storitve, čakalne vrste opravil in UI za nadzor eksperimentov.

---

## 1) PRD – Product Requirements Document

### 1.1 Vizija
Poenostaviti fine‑tuning LLM/ML modelov za podatkovne znanstvenike in inženirje preko intuitivnega vmesnika, ki generira reproducibilne noteboooke z najboljšimi praksami.

### 1.2 Ciljni uporabniki
- ML inženirji, podatkovni znanstveniki, AI svetovalci.
- Napredni analitiki (želijo hitre iteracije brez ročnega lepljenja skript).

### 1.3 Ključne vrednosti
- **Hitrost do prvega rezultata** (ready‑to‑run notebook).  
- **Reproducibilnost** (fiksni seed, beleženje konfiguracije).  
- **Varnost & skladnost** (lokalna obdelava, jasna pravila),  
- **Razširljivost** (pluggable providerji/modeli).

### 1.4 Uspešnostni KPI‑ji (MVP/Faza 1)
- < 10 min do generiranega in veljavnega Notebooka.  
- ≥ 90% uspešnost "smoke" treninga na vzorčnem datasetu.  
- ≥ 80% uporabnikov oceni UI za parametre kot “jasen”.

### 1.5 Out‑of‑Scope (Faza 1)
- Dolgotrajni backend job runnerji, multi‑tenant kvote, napredni RBAC, plačila, orkestracija grozdov.

---

## 2) SRS – Software Requirements Specification

### 2.1 Funkcionalne zahteve (Faza 1)
1. **Uvoz parametrov**: UI obrazci za: izvorni model, vrsto fine‑tuninga (poln/LoRA/QLoRA), hiperparametre, poti do datasetov, eval metrike, nastavitve HW (CPU/GPU), reproducibilnost (seed).  
2. **Validacija**: preverjanje tipov, obveznih polj, kompatibilnosti (npr. QLoRA zahteva 4‑bit quant + podporo).  
3. **Generacija Notebooka**: iz predloge + parametri → `.ipynb` z jasnimi celicami: namestitev, validacija, priprava podatkov, trening, eval, shranjevanje artefaktov, inferenca demo.  
4. **Prenos**: prenos `.ipynb` datoteke in sidecar `config.yaml`.  
5. **Audit trail**: v Notebook vpiše celoten `config` (markdown + koda), verzije knjižnic, hash datasetov (če možno).  
6. **Testni vzorčni dataset**: možnost auto‑generacije mini datasetov za smoke test (npr. 50 vrstic).

### 2.2 Ne‑funkcionalne zahteve
- **Uporabnost**: obrazci z inline pomočjo, prednastavljene vrednosti za začetnike.  
- **Zanesljivost**: 100% deterministična generacija Notebooka ob enakem vnosu.  
- **Varnost**: podatki ostanejo lokalno v brskalniku ali se nalagajo le za validacijo (brez shranjevanja na strežnik v Fazi 1).  
- **Kompatibilnost**: Notebook deluje v lokalnem Jupyterju in Google Colabu.  
- **Dokumentiranost**: auto‑generiran `README.md` v Notebooku.

### 2.3 Omejitve in odvisnosti
- Dostopnost osnovnih modelov (HF Hub, lokalni checkpointi).  
- Strojna oprema (GPU priporočena za trening).  
- Licenčne omejitve posameznih modelov/datasetov.

---

## 3) Uporabniški vlogi in dovoljenja (Faza 1)
- **Urednik**: izpolni obrazce, generira notebook, prenese datoteke.  
- (Kasneje) **Admin**: upravljanje providerjev, ključi, kvote.  
- (Kasneje) **Viewer**: dostop do poročil/artefaktov.

---

## 4) User Stories (Faza 1)
1. Kot uporabnik želim izbrati **osnovni model** in **tip fine‑tuninga**, da dobim notebook z ustreznim kôdom.  
2. Kot uporabnik želim **naložiti ali referencirati dataset** (lokalna pot ali HF id), da ga notebook samodejno pripravi.  
3. Kot uporabnik želim nastaviti **hiperparametre** (lr, batch, epoch, warmup …) in dobiti validirane predloge.  
4. Kot uporabnik želim izbrati **metrike evalvacije** (perplexity, ROUGE, accuracy), da vidim rezultate po treningu.  
5. Kot uporabnik želim klikniti **Prenesi**, da dobim `.ipynb` + `config.yaml` za reproducibilnost.

---

## 5) Parametri (UI ↔ Notebook)

### 5.1 Osnovni model
- Provider: `hf_local | hf_hub | openai_ft | vllm_local` (Faza 1: `hf_local` in `hf_hub`).  
- Model id: npr. `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, ipd.  
- Naloži tokenizer z istim id + varnostne zastavice (trust_remote_code).

### 5.2 Tip fine‑tuninga
- `full` (klasični FT), `lora` (PEFT), `qlora` (4‑bit quant + LoRA).
  - LoRA polja: `r`, `alpha`, `dropout`, `target_modules` (prednastavljeno), `bias`.  
  - QLoRA polja: `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`, `bnb_4bit_compute_dtype`.

### 5.3 Hiperparametri
- `learning_rate`, `batch_size_train`, `batch_size_eval`, `num_epochs`, `gradient_accumulation`, `weight_decay`, `lr_scheduler`, `warmup_ratio`, `max_seq_len`, `seed`.

### 5.4 Dataset
- Vir: `upload_local_path | hf_dataset_id`.  
- Format: `jsonl_chat | jsonl_instr | csv_classification`.  
- Delitev: `train/val/test` razmerja ali pot do ločenih datotek.  
- Validacija: preveri obvezna polja in nizko stopnjo praznin/duplikatov.

### 5.5 Eval & logiranje
- Metrike: `perplexity`, `rouge`, `accuracy`, `f1` (odvisno od tipa naloge).  
- Sledenje: `wandb | tensorboard | none` (Faza 1: `tensorboard | none`).

### 5.6 HW
- `device`: `auto | cpu | cuda` (Notebook samodejno zazna).  
- `mixed_precision`: `fp16 | bf16 | none`.

### 5.7 Artefakti
- `output_dir`, opcija `push_to_hub` (privzeto off v Fazi 1), `save_strategy`, `save_total_limit`.

---

## 6) Format datasetov (primeri)

### 6.1 `jsonl_chat`
Vsaka vrstica JSON z obliko:
```json
{"messages": [
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Vprašanje"},
  {"role": "assistant", "content": "Odgovor"}
]}
```

### 6.2 `jsonl_instr`
```json
{"instruction": "Prevedi…", "input": "…", "output": "…"}
```

### 6.3 `csv_classification`
Stolpci: `text,label` (header obvezen). Labeli enoviti nizi.

---

## 7) Arhitektura (visokonivojsko)

**Faza 1 (MVP)**
- Frontend SPA (npr. React/Next.js ali klasičen Vite React): obrazci → validacija → generacija `.ipynb` in `config.yaml` lokalno v brskalniku (File API) ali preko lahkega Node/py backend končnega pointa `/generate_notebook`.
- Predloge (templates) za Notebook in YAML vgrajene v aplikacijo.

**Kasnejše faze**
- Backend (FastAPI) + Worker (Celery/Arq) za asinkrono treniranje, HL orchestrator, artefakti v objektni hrambi, UI za sledenje eksperimentom.

---

## 8) API dizajn (Faza 1 minimal)

`POST /generate_notebook`
- **Body**: `application/json` – popoln objekt konfiguracije (glej 5.*).  
- **Response**: `application/zip` ali `multipart` z `notebook.ipynb` in `config.yaml`.

Validacijske napake → `400` z opisom polja.

(OpenAPI specifikacija bo generirana v Fazi 2.)

---

## 9) UI/UX (Faza 1)

### 9.1 Strani
1. **Wizard** (4 koraki):
   - Korak 1: Izbira modela (provider, model id, tip FT) + inline info.  
   - Korak 2: Dataset (vir, format, delitev, predogled 20 vrstic, validacija).  
   - Korak 3: Hiperparametri & HW (prednastavitve + napredne nastavitve).  
   - Korak 4: Povzetek & Generiraj (prikaz `config`, gumb Prenesi).

2. **Pomoč**: vodiči za formate in primere.

### 9.2 Komponente
- Vnos model id + gumb “Preveri razpoložljivost”.  
- Nalagalnik datotek z validacijo velikosti/končnice.  
- Tabela za predogled.  
- Polja z enotami (npr. `learning_rate` z validacijskim razponom).  
- Badge za opozorila (npr. QLoRA potrebuje 4‑bit quant).  
- JSON/YAML viewer s kopiranjem.

---

## 10) Notebook: struktura in vsebina

1. **Naslov & povzetek** (Markdown): opis eksperimenta + tabela parametrov.  
2. **Namestitev odvisnosti**: `pip install` (transformers, datasets, peft, bitsandbytes, accelerate, evaluate, tensorboard, trl po potrebi).  
3. **Uvoz knjižnic in seed**.  
4. **Nalagalni blok konfiguracije**: auto‑vstavljeni `CONFIG` (Python dict ali YAML → dict).  
5. **Validacija konfiguracije** (asserti, human‑friendly napake).  
6. **Priprava podatkov**: branje, čiščenje, delitev, tokenizacija (pad/truncate na `max_seq_len`).  
7. **Nalaganje modela & tokenizerja**: full/LoRA/QLoRA vejitev.  
8. **Trening**: `Trainer` ali `TRL SFTTrainer` (odvisno od formata), z `TrainingArguments`.  
9. **Evalvacija**: izračun metrik glede na tip naloge.  
10. **Shranjevanje artefaktov**: `output_dir`, model card (`README.md`), shranjevanje LoRA adapterjev.  
11. **Inferenca demo**: nekaj primerov s časom izvajanja.  
12. **Dnevnik okolja**: verzije paketov, CUDA info.  
13. **Smoke test celica**: hitro preverjanje na mini vzorcu.

---

## 11) Testni načrt (Faza 1)

### 11.1 Vrste testov
- **Validacijski testi UI**: nepravilni tipi/razponi (npr. negativen `learning_rate`).  
- **Notebook smoke test**: 1 epoch na mini datasetu (< 2 min CPU/GPU).  
- **Determinističnost**: enak seed → enaka začetna metrika (znotraj tolerance).  
- **Kompatibilnost**: Notebook se zažene v lokalnem Jupyterju in v Colabu (brez ročnih popravkov).

### 11.2 Primeri kriterijev sprejema (per korak)
- **K1 – Validacija obrazca**: vse obvezne vrednosti izpolnjene; napačna vrednost sproži jasno sporočilo.  
- **K2 – Generacija datotek**: ustvarjen veljaven `.ipynb` (nbformat OK) + `config.yaml`.  
- **K3 – Zagon**: `pip install` deluje, GPU auto‑detect ne prekine procesa na CPU.  
- **K4 – Smoke trening**: izvede se 1 epoch, ni runtime napak; zabeležen TensorBoard log.  
- **K5 – Artefakti**: `output_dir` vsebuje adapterje/model in `README.md`.

---

## 12) Načrt izvedbe po korakih (fazah) z zadržki (gates)

**Faza 1.0 – Spec & predloge**  
- Naloge: PRD/SRS, parametri, predloge Notebook/YAML, UX wire.  
- Test: pregled skladnosti; pilot validacija parametrov.  
- DoD: odobreni dokumenti + primer `config.yaml`.

**Faza 1.1 – UI obrazci + validacija**  
- Naloge: implementacija obrazcev, shema validacije (Zod/Yup), prednastavitve.  
- Test: enote testov za 20 validacijskih scenarijev.  
- Gate: QA potrdi 0 blockerjev.

**Faza 1.2 – Generator Notebooka**  
- Naloge: templating (Jinja2), vstavljanje parametrov, tvorba `.ipynb` (nbformat).  
- Test: nbformat validate; snapshot test na 3 konfiguracijah.

**Faza 1.3 – Prenos & sidecar**  
- Naloge: ZIP/multipart prenos, metapodatki, checksum.  
- Test: datoteke se odprejo in zagnjejo brez napak.

**Faza 1.4 – QA smoke run**  
- Naloge: zagon v Colabu + lokalno, mini dataset, GPU/CPU.  
- Test: končana 1 epoha, logi ustvarjeni, artefakti prisotni.  
- Gate: podpis PO.

---

## 13) Repo struktura (predlog)
```
repo/
  app/                  # frontend (Faza 1 lahko brez backenda)
  templates/
    notebook.tpl.json   # Jinja2 nbformat predloga
    readme.tpl.md
    config.tpl.yaml
  schemas/
    config.schema.json
  examples/
    configs/
    datasets/
  tests/
    unit/
    snapshot/
  docs/
    PRD.md
    SRS.md
    TESTPLAN.md
```

---

## 14) Varnost & skladnost
- Brez nalaganja občutljivih podatkov na strežnik v Fazi 1.  
- Jasna opozorila glede licenc modelov in datasetov.  
- Opt‑in telemetrija (anonimna) v kasnejših fazah.

---

## 15) Opazljivost (kasneje)
- Integracija z TensorBoard/W&B; zbiranje metrik po eksperimentih.

---

## 16) Tveganja & omilitve
- **Nezdružljivosti paketov** → pin različic, preverjanje v Notebooku.  
- **GPU omejitve** → QLoRA + možnost CPU demo/smoke.  
- **Format datasetov** → stroga validacija + primeri.

---

## 17) Traceability matrika (izsek)
| User Story | Zahteva | Test |
|---|---|---|
| US‑1 | Izbira modela | K1, K2 |
| US‑2 | Dataset uvoz | K1, K4 |
| US‑3 | Hiperparametri | K1 |
| US‑4 | Eval metrike | K4 |
| US‑5 | Prenos datotek | K2, K3 |

---

## 18) “Prompt Pack” za **Codex** (navodila za generiranje kode)

### 18.1 Sistem (stalna pravila)
- Piši produkcijsko kodo, prednost ima jasnost in robustnost.  
- Uporabi **nbformat** za ustvarjanje `.ipynb` iz predloge.  
- Validiraj vhodni `config` proti `schemas/config.schema.json`.  
- Ne dostopaj do interneta; ne vgrajuj API ključev.  
- Rezultat naj bo determinističen pri danem `seed`.

### 18.2 Naloga: Generator Notebooka (Faza 1)
**Vhod**: JSON `config` (glej §5)  
**Izhod**: datoteka `fine_tune.ipynb` + `config.yaml` + `README.md`

**Koraki, ki jih mora koda implementirati:**
1. V brskalniku/na strežniku zgradi Notebook iz `templates/notebook.tpl.json` in Jinja2 polj.  
2. Vstavi poln `CONFIG` v prvo kodo celico.  
3. Zagotovi celice za: namestitev, validacijo, pripravo podatkov, trening, eval, artefakte, demo.  
4. Dodaj "smoke test" celico, ki teče v < 2 minutah na CPU.  
5. Shrani `README.md` iz predloge z opisom parametrov in kako zagnati Notebook.  
6. Ustvari ZIP z vsebino za prenos.

**Merila sprejema (Codex):**
- `nbformat.validate` brez napak.  
- Vstavljen `CONFIG` se ujema z vhodnim JSON.  
- Notebook se požene v Colabu brez ročnega posega (razen odobritev) in v lokalnem Jupyterju.

### 18.3 Vzorčni “user” prompt za Codex
> "Ustvari Node/TypeScript modul `generateNotebook.ts`, ki sprejme `config: Config`, validira ga proti `schemas/config.schema.json` in iz `templates/` zgradi `fine_tune.ipynb`. Uporabi `nbformat` (prek `notebookjs` ali Python skrita naloga, če Node knjižnica manjka – generiraj tudi Python skript). Priloži enote teste (Vitest) za 3 različne konfiguracije (full, lora, qlora) in snapshot test za notebook. Ne uporabljaj zunanjih storitev. Konfiguriraj `pnpm` skripte in `tsup` build."

---

## 19) Kodni standardi & CI
- Linters: `eslint` (TS), `ruff/black` (py templates).  
- Testing: `vitest` (TS), `pytest` (notebook utili).  
- CI: ob PR poženi validacije shem, unit & snapshot teste, preveri `nbformat.validate`.

---

## 20) Priloge
- **Primer `config.yaml`** (izsek):
```yaml
provider: hf_hub
model_id: mistralai/Mistral-7B-Instruct-v0.3
tune_type: qlora
hyperparams:
  learning_rate: 2.0e-4
  num_epochs: 1
  batch_size_train: 8
  gradient_accumulation: 2
  warmup_ratio: 0.03
  max_seq_len: 2048
seed: 42
dataset:
  source: hf_dataset_id
  id: samsum
  format: jsonl_instr
  split: {train: 0.98, val: 0.02}
eval:
  metrics: [rouge]
logs: tensorboard
hw:
  device: auto
  mixed_precision: bf16
artifacts:
  output_dir: outputs/run_001
```

- **Kontrolni seznam (per korak)**:  
  □ Shema validacije posodobljena  
  □ Snapshoti Notebooka osveženi  
  □ Smoke test na CPU  
  □ README generiran  
  □ Prenos ZIP deluje

---

## 21) Backlog za naslednje verzije
- Faza 2: Backend (FastAPI), čakalna vrsta (Celery/Redis), shranjevanje artefaktov, dashboard eksperimentov.  
- Faza 3: Multi‑provider (OpenAI FT, Bedrock), RBAC, organizacijski prostori.  
- Faza 4: Auto‑eval (crowd/LLM‑as‑judge), primerjava eksperimentov, poročila PDF.

---

**Konec dokumenta**

