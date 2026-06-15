# Roman Disperser Tutorials — NERSC conda environment

Two shared environments are pre-built on NERSC — `tutorial_2026_cpu` and
`tutorial_2026_gpu` — and `.grism_sim_setup` exports their paths plus
`ROMAN_DISPERSER_DATA` (the shared reference data). You just activate one.

## Shell use

```bash
source /global/common/software/m4943/.grism_sim_setup
module load conda
conda activate $tutorial_2026_cpu     # or $tutorial_2026_gpu  (on GPU nodes)
```

## Notebooks — first-time kernel setup

Register the Jupyter kernels (once per user). This installs **both** the CPU and
GPU kernels, derives the names automatically from the active env, and wraps each
in `kernel-helper.sh` (which sets up the reference-data env, including
`ROMAN_DISPERSER_DATA`):

```bash
source /global/common/software/m4943/.grism_sim_setup
module load conda

for V in cpu gpu; do
    envvar="tutorial_2026_$V"
    conda activate "${!envvar}"
    KNAME=$(basename "$CONDA_PREFIX")                 # roman-tutorials-cpu / -gpu
    python -m ipykernel install --user \
        --name "$KNAME" \
        --display-name "Roman Disperser Tutorials ($V)"
    sed -i '/"argv": \[/a\  "/global/common/software/m4943/kernel-helper.sh",' \
        "$HOME/.local/share/jupyter/kernels/$KNAME/kernel.json"
done
```

In the Jupyter launcher you'll then see **Roman Disperser Tutorials (cpu)** and
**(gpu)**. Use the GPU kernel only in a GPU-node Jupyter session.

## Verify — in a notebook cell (not just the terminal)

A kernel does **not** inherit your shell, so confirm the kernel itself sees the
data. Open a notebook on the kernel and run:

```python
import jax
from roman_disperser.pipeline import resolve_paths
from roman_disperser.optical_model import RomanOpticalModel
*_, om, _ = resolve_paths()
RomanOpticalModel(str(om))                      # fails if the kernel can't find the data
print("data at", om.parent, "; backend:", jax.default_backend())
```

Expect `backend: cpu` on the CPU kernel, `gpu` on a GPU node.

> **If you get a `FileNotFoundError`** the kernel isn't receiving
> `ROMAN_DISPERSER_DATA`. `kernel-helper.sh` must provide it: if the helper
> `source`s `.grism_sim_setup` it's automatic; otherwise add
> `export ROMAN_DISPERSER_DATA=…` to the helper, or add an `"env"` block to each
> `kernel.json`:
> `"env": {"ROMAN_DISPERSER_DATA": "/dvs_ro/cfs/cdirs/m4943/grismsim/roman-disperser-data"}`.
