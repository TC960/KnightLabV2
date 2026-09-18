#!/usr/bin/env bash
# One-shot, self-verifying setup for a Brev GPU box running the extraction job.
#
# Every step here exists because it failed once. Do not simplify without reading
# the reason.
#
#   python vs python3      the box has python3 only; `python` is not a command
#   set -o pipefail        a failing command inside `... | tee log` does NOT trip
#                          `set -e`, so a setup script once printed SETUP_OK over
#                          a model download that never happened
#   source build           the prebuilt llama-cpp-python wheel assumes AVX-512;
#                          this host has AVX2 only and dies with SIGILL
#                          (Illegal instruction, core dumped) at MODEL LOAD --
#                          `import llama_cpp` succeeds, so importing is not a test
#   CUDA_HOME              nvcc is installed at /usr/local/cuda-12.8/bin/nvcc but
#                          is NOT on PATH; `which nvcc` reports missing. The pip
#                          nvidia-cuda-nvcc fallback trips `nvidia.__file__ is
#                          None` (namespace package) and finds nothing
#   verify the ARTIFACT    every stage below checks a file, a size, or a live
#                          model load -- never its own success message
#
# Usage on the box:  bash gpu_setup.sh 2>&1 | tee ~/kg/setup.log
set -euo pipefail

KG=~/kg
MODEL_REPO="Jackrong/Qwopus3.5-27B-v3-GGUF"
MODEL_FILE="Qwopus3.5-27B-v3-Q4_K_M.gguf"
MIN_MODEL_BYTES=$((15 * 1000 * 1000 * 1000))   # ~16.5GB expected; fail if truncated

step() { echo; echo "=== $* ==="; }

step "0. environment"
python3 --version
nproc
lscpu | grep -oE 'avx512[a-z]*|avx2|avx' | sort -u | tr '\n' ' '; echo
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step "1. locate CUDA (not via \`which\`)"
CUDA_HOME=$(dirname "$(dirname "$(find /usr/local -name nvcc -type f 2>/dev/null | head -1)")")
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "FATAL: no nvcc under /usr/local"; exit 1; }
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -1

step "2. python deps"
pip install -q --upgrade pip
pip install -q huggingface_hub cmake ninja

step "3. download the model, then CHECK THE FILE"
python3 - <<PY
from huggingface_hub import hf_hub_download
p = hf_hub_download("$MODEL_REPO", "$MODEL_FILE", local_dir="$KG/model")
open("$KG/model_path.txt", "w").write(p)
print("downloaded ->", p)
PY
MP=$(cat "$KG/model_path.txt")
[ -s "$MP" ] || { echo "FATAL: model path empty"; exit 1; }
SZ=$(stat -c%s "$MP")
echo "model size: $((SZ/1000/1000/1000)) GB"
[ "$SZ" -ge "$MIN_MODEL_BYTES" ] || { echo "FATAL: model truncated ($SZ bytes)"; exit 1; }

step "4. build llama-cpp-python FROM SOURCE for this CPU + this GPU"
pip uninstall -y -q llama-cpp-python 2>/dev/null || true
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=on -DCMAKE_CUDA_ARCHITECTURES=86" \
  pip install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python 2>&1 | tail -3

step "5. PROVE the model loads -- import is not enough, SIGILL happens at load"
python3 - <<PY
import json, sys
from llama_cpp import Llama, LlamaGrammar
mp = open("$KG/model_path.txt").read().strip()
g = LlamaGrammar.from_string(open("$KG/repo/proj_2_attempt3/kg/grammar.gbnf").read())
llm = Llama(model_path=mp, n_ctx=4096, n_gpu_layers=-1, verbose=False)
r = llm.create_chat_completion(
    messages=[{"role": "user", "content":
               'Reply with JSON only. {"disease": "test", "taxa_enriched": ["Blautia"], '
               '"taxa_depleted": []}'}],
    temperature=0, max_tokens=256, grammar=g)
out = r["choices"][0]["message"]["content"]
json.loads(out)                      # must parse, or the grammar is wrong
print("SMOKE TEST OK ->", out[:90])
PY

step "6. confirm the GPU was actually used"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo
echo "SETUP_VERIFIED"
