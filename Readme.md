# Compact SAM Decoding (C-SAMD)

This repository is the reference implementation of **Compact SAM Decoding:
Space-Efficient Suffix Automata for Retrieval-Based Speculative Decoding**.
C-SAMD keeps the online retrieval and target-model verification procedure of
SAM Decoding (SAMD), while replacing the large static suffix-automaton index
with:

- a typed C++ implementation;
- generalized suffix-automaton (GSAM) construction that does not create
  cross-document substrings;
- Open Addressing (OA) transition tables; and
- lazy inline storage for low-degree transition sets.

The implementation is based on the public
[Spec-Bench](https://github.com/hemingkx/Spec-Bench) evaluation framework and
[SAM-Decoding](https://github.com/hyx1999/SAM-Decoding). This README describes
the complete reproduction path.

## Repository layout

| Path | Purpose |
| --- | --- |
| `model/gsamd/` | C-SAMD implementation and C++ extension |
| `model/gsamd/cpp/` | GSAM/SAM construction and transition containers |
| `model/gsamd/tools/build_static_sam.py` | Build a serialized C-SAMD static index |
| `evaluation/inference_gsamd.py` | C-SAMD Spec-Bench entry point |
| `eval_7b.sh` / `eval_13b.sh` | Main, ablation, and hybrid experiments |
| `eval_percent.sh` | Static-corpus scaling experiment |
| `scripts/run_dynamic_only.sh` | Three-run experiment without an external corpus |
| `scripts/get_speed_table.py` | Convert raw JSONL answers into per-task CSV metrics |
| `scripts/generate_experiment_artifacts.py` | Aggregate repeated runs and generate paper tables/figures |
| `tests/test_gsamd.py` | Construction, serialization, backend, and decoding tests |

The other directories retain comparison methods from Spec-Bench so that all
systems can be measured in one evaluation framework.

## Reproduction environment

The reported experiments used:

- Ubuntu 22.04, Linux 5.15, x86-64;
- Python 3.12.12;
- GCC/G++ 11.4.0 with C++17;
- PyTorch 2.5.1 with CUDA 12.1 and cuDNN 9.1;
- Transformers 4.37.1;
- one NVIDIA RTX 4090 with 24 GB VRAM for each 7B run;
- two NVIDIA RTX 4090 GPUs for the supplementary 13B run;
- NVIDIA driver 550.67; and
- two Intel Xeon Platinum 8488C sockets with 192 logical CPU threads and
  1.08 TB of host RAM.

Timing results depend on the GPU, CPU, driver, thermal state, and other
processes on the host. Run the autoregressive baseline and the speculative
method on the same machine and in the same run directory. The paper reports
the arithmetic mean and sample standard deviation of three independent
end-to-end runs unless explicitly marked as a single run.

### Installation

```bash
cd GSAM-Decoding

conda create -n csamd python=3.12 -y
conda activate csamd
pip install -r requirements.txt

python model/gsamd/setup.py build_ext --inplace
pytest -q tests/test_gsamd.py
```

`protobuf==3.19.0` and `pybind11==3.0.1` are intentional compatibility pins.
If PyTorch must be installed from a CUDA-specific channel on your machine,
install PyTorch 2.5.1 first and then install the remaining requirements.

Download the target model from its public Hugging Face repository:

```bash
git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3 /path/to/models/vicuna-7b-v1.3
```

The 13B check uses
[lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3).
The hybrid experiment additionally uses
[yuhuili/EAGLE-Vicuna-7B-v1.3](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3).

## Benchmark data and leakage control

The tracked file `data/spec_bench/question.jsonl` is the public Spec-Bench test
set. Make three run directories before a repeated experiment:

```bash
for run in 1 2 3; do
  mkdir -p "data/spec_bench_7b_${run}/model_answer"
  cp data/spec_bench/question.jsonl "data/spec_bench_7b_${run}/question.jsonl"
done
```

The static SAM corpus and Spec-Bench both involve GSM8K, but they use disjoint
official splits:

- static-corpus construction reads the 7,473-example GSM8K **train** split;
- Spec-Bench mathematical reasoning uses 80 examples from the GSM8K **test**
  split; and
- none of those 80 test questions occurs in the training split.

The split choice can be checked in the public SAM-Decoding corpus-preparation
script and the
[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) repository.
It can also be verified locally after downloading GSM8K by matching the 80
`math_reasoning` questions in `data/spec_bench/question.jsonl` against the
official `train` and `test` columns.

## Build the static corpus and index from scratch

The paper follows the public SAM-Decoding corpus recipe. It combines prompts
from:

- [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned);
- the **train** split of
  [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k); and
- [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca).

Prompts are formatted with the Vicuna template. Vicuna-7B-v1.3 produces the
corpus responses with temperature 0.8, top-p 0.95, at most 1,024 new
tokens, and vLLM seed 0. These are the settings in the public SAM-Decoding workflow.

First construct the prompt/response dataset:

```bash
git clone https://github.com/hyx1999/SAM-Decoding.git /path/to/SAM-Decoding
cd /path/to/SAM-Decoding

git clone https://huggingface.co/datasets/yahma/alpaca-cleaned sam_data/alpaca-cleaned
git clone https://huggingface.co/datasets/openai/gsm8k sam_data/gsm8k
git clone https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca \
  sam_data/python_code_instructions_18k_alpaca

python -m tools.prepare_prompts \
  --model_name /path/to/models/vicuna-7b-v1.3 \
  --cutoff_len 1024 \
  --prompt_template_name vicuna

CUDA_VISIBLE_DEVICES=0 python -m tools.gen_response \
  --model_name /path/to/models/vicuna-7b-v1.3 \
  --sam_data_path sam_data/sam_prompts
```

The last command uses vLLM, as specified by SAM-Decoding. The original corpus
workflow used `vllm==0.2.7` in a Python 3.11 environment. Corpus generation is
an offline step and may be performed in a separate environment from C-SAMD.
Keep the resulting `sam_data/sam_dialogues` directory.

Then build the full C-SAMD index:

```bash
cd /path/to/GSAM-Decoding

python model/gsamd/tools/build_static_sam.py \
  --tokenizer_path /path/to/models/vicuna-7b-v1.3 \
  --sam_data_path /path/to/SAM-Decoding/sam_data/sam_dialogues \
  --sam_path local_cache/gsamd \
  --cutoff_len 2048 \
  --n_predicts 40 \
  --use_gsam \
  --map_type lazy_int32 \
  --lazy_threshold 1
```

This produces
`local_cache/gsamd/static_data_gsam_lazy_suffix_int32_t1.pb`. To construct the
logical SAM rather than GSAM control, add `--no-use_gsam`. To build the full
transition-container ablation, add `--build_map_ablation`; the same serialized
logical automaton may also be loaded with a different runtime map using
`--samd_map_type` and `--samd_lazy_threshold`.

The original Python SAMD baseline index can be constructed from the same
`sam_dialogues` dataset with the public SAM-Decoding
`tools/gen_sam_alpaca_sam_only.py` script. Prebuilt indexes are deliberately not linked here; use the from-scratch procedure
above.

### Measure index construction cost

The build-cost table was measured separately with GNU `/usr/bin/time`; it is
not generated from the Spec-Bench CSV files. Run each configuration in five
fresh processes with the same tokenizer, dialogue corpus, and machine:

```bash
mkdir -p logs/build_cost local_cache/build_cost
for run in 1 2 3 4 5; do
  /usr/bin/time -v -o "logs/build_cost/sam_stl_${run}.time" \
    python model/gsamd/tools/build_static_sam.py \
      --tokenizer_path /path/to/models/vicuna-7b-v1.3 \
      --sam_data_path /path/to/SAM-Decoding/sam_data/sam_dialogues \
      --sam_path "local_cache/build_cost/sam_stl_${run}" \
      --cutoff_len 2048 --n_predicts 40 \
      --no-use_gsam --map_type unordered

  /usr/bin/time -v -o "logs/build_cost/gsam_lazy_oa_${run}.time" \
    python model/gsamd/tools/build_static_sam.py \
      --tokenizer_path /path/to/models/vicuna-7b-v1.3 \
      --sam_data_path /path/to/SAM-Decoding/sam_data/sam_dialogues \
      --sam_path "local_cache/build_cost/gsam_lazy_oa_${run}" \
      --cutoff_len 2048 --n_predicts 40 \
      --use_gsam --map_type lazy_int32 --lazy_threshold 1
done
```

Use `Elapsed (wall clock) time` as build time and `Maximum resident set size`
as peak RSS. On Linux, convert the reported KiB value to decimal GB by
multiplying by 1,024 and dividing by 1,000,000,000. Report the five-run
arithmetic mean and sample standard deviation. The paper table reports
69.19 +/- 0.42 s and 13.84 GB for SAM + STL, and 55.13 +/- 0.68 s and
7.35 GB for GSAM + lazy + OA. Corpus generation and model inference are not
included.

### Reproduce the corpus-scaling subsets

The 1%, 5%, 10%, and 50% corpora are nested prefixes of one shuffle with seed
42; the 100% condition uses the complete `sam_dialogues` dataset. Create the
subsets once, then reuse the same subsets in all three timing runs:

```bash
python scripts/sample_sam_dialogues.py \
  --input-path /path/to/SAM-Decoding/sam_data/sam_dialogues \
  --output-root /path/to/corpus-subsets \
  --percentages 1 5 10 50 \
  --seed 42
```

Build each subset with the same `build_static_sam.py` options shown above,
changing only `--sam_data_path` and `--sam_path`. Evaluate every resulting
index against the autoregressive baseline in each of the three
`spec_bench_7b_percent_<run>` directories. This keeps corpus membership fixed
across repetitions and varies only the end-to-end timing run.

## Run C-SAMD

The following command evaluates the complete 7B C-SAMD configuration on one
Spec-Bench run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python -m evaluation.inference_gsamd \
  --model-path /path/to/models/vicuna-7b-v1.3 \
  --model-id vicuna-7b-v1.3-csamd \
  --bench-name spec_bench_7b_1 \
  --temperature 0.0 \
  --dtype float16 \
  --samd_n_predicts 40 \
  --samd_len_threshold 5 \
  --samd_len_bias 5 \
  --attn_implementation sdpa \
  --static_sam_path local_cache/gsamd/static_data_gsam_lazy_suffix_int32_t1.pb \
  --samd_map_type lazy_int32 \
  --samd_lazy_threshold 1
```

Run the autoregressive baseline in the same directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_baseline \
  --model-path /path/to/models/vicuna-7b-v1.3 \
  --model-id vicuna-7b-v1.3-vanilla-float16-temp-0.0 \
  --bench-name spec_bench_7b_1 \
  --temperature 0.0 \
  --dtype float16
```

Repeat both commands with `spec_bench_7b_2` and `spec_bench_7b_3`. The
repository's `eval_7b.sh` runs the paper's complete comparison and ablation
matrix; set its model/cache paths before use. `eval_13b.sh` contains the
supplementary two-GPU configuration.

### Dynamic-SAM-only control

Omitting `--static_sam_path` disables all external-corpus retrieval. C-SAMD
then drafts only from the dynamic SAM built from the current request. The
three-run launcher reuses an existing autoregressive result in each directory:

```bash
MODEL_PATH=/path/to/models/vicuna-7b-v1.3 \
GPU_IDS="0 1 2" \
bash scripts/run_dynamic_only.sh
```

Each of the following directories must already contain
`vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl`:

```text
data/spec_bench_7b_percent_1/model_answer/
data/spec_bench_7b_percent_2/model_answer/
data/spec_bench_7b_percent_3/model_answer/
```

The launcher writes `vicuna-7b-v1.3-csamd-dynamic-only.jsonl` beside each
baseline. No static index is loaded or queried.

## Compute metrics

Convert every run's raw answers to a CSV:

```bash
for run in 1 2 3; do
  python scripts/get_speed_table.py \
    --dir-path "data/spec_bench_7b_${run}/model_answer" \
    --csv-path "data/spec_bench_7b_${run}/result.csv" \
    --tokenizer-path /path/to/models/vicuna-7b-v1.3
done
```

For the dynamic-only control, substitute `spec_bench_7b_percent_${run}`.
`scripts/get_speed_table.py` computes:

- per-example generated tokens per second;
- speedup as the mean speculative throughput divided by the mean
  autoregressive throughput in the same run;
- accepted draft tokens per verification step; and
- host RSS, stored as the mean of 2-second samples for each question; the CSV
  reports the maximum per-question value.

Generate the paper tables and figures from raw CSV files:

```bash
python scripts/generate_experiment_artifacts.py \
  --root /path/to/paper/source \
  --result-7b \
    data/spec_bench_7b_1/result.csv \
    data/spec_bench_7b_2/result.csv \
    data/spec_bench_7b_3/result.csv \
  --dynamic-only-results \
    data/spec_bench_7b_percent_1/result.csv \
    data/spec_bench_7b_percent_2/result.csv \
    data/spec_bench_7b_percent_3/result.csv \
  --result-13b data/spec_bench_13b/result.csv \
  --degree-stats data/index_degree_stats.csv
```

The command writes every table in the paper `generated/` directory. The cached
`data/index_degree_stats.csv` contains state-count statistics derived from the
serialized SAM, GSAM, and corpus-scaling indexes. Refresh it only after
rebuilding those indexes:

```bash
python scripts/generate_experiment_artifacts.py \
  --root /path/to/paper/source \
  --refresh-degree-stats \
  --tables-only
```

The default refresh inputs are `local_cache/gsamd/static_data_sam.pb` and
`static_data_gsam_{1,5,10,50}.pb` plus `static_data_gsam.pb`. Override a path
with repeated `--degree-index NAME=PATH`, where `NAME` is `sam-100` or one of
`gsam-1`, `gsam-5`, `gsam-10`, `gsam-50`, and `gsam-100`. A normal cached
`--tables-only` run completes without loading the multi-GB protobuf files. The
manually timed build-cost table is intentionally embedded in the manuscript
and is the only paper table not produced by this script.

The script uses sample standard deviation (`ddof=1`) across the three runs.
Raw JSONL, per-run CSV, and the exact command line should be retained when
reporting results.

## Expected headline results

On the paper machine, the full 7B C-SAMD index uses 6.75 decimal GB of host
RSS, compared with 35.22 GB for the original Python SAMD index. The complete
C-SAMD configuration obtains approximately 1.80x overall speedup on
Spec-Bench. With the static corpus disabled, dynamic-only C-SAMD uses 1.33 GB,
obtains 1.667x +/- 0.011 speedup, and accepts 1.770 tokens per verification.
These values are hardware- and corpus-dependent; reproduce the
same qualitative comparison rather than expecting bit-identical timing.

## Correctness notes

- The target-model verification rule is unchanged. Different speculative
  execution paths can nevertheless change FP16 rounding, so byte-for-byte
  greedy outputs are not guaranteed; use `evaluation/equal.py` when exact
  equality with a saved baseline is required.
- GSAM deliberately removes internal-EOS and cross-document paths. It can
  change proposed candidates, accepted length, and runtime.
- Container-only variants represent the same transition function and are
  covered by backend-equivalence and serialization tests.
- Memory values are decimal-GB host RSS for the process. They are not GPU
  memory and do not include a claim that model weights or KV cache are smaller.

## Acknowledgments

This repository retains components from
[Spec-Bench](https://github.com/hemingkx/Spec-Bench),
[SAM-Decoding](https://github.com/hyx1999/SAM-Decoding), and the comparison
methods integrated by those projects. See `LICENSE` and the upstream
repositories for their respective terms.
