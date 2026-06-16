# Setup

These tutorials use the [`roman_disperser`](https://github.com/roman-grs-pit/roman_disperser)
library. Pick the path that matches where you're running; both end the same way:
a Jupyter kernel that can `import roman_disperser` and find its reference data.
When you're set up, **notebook `00_environment_check.ipynb` verifies everything**
— run it first.

> **One JAX rule for everyone:** `roman_disperser` is JAX-based. The notebooks
> run on **CPU or GPU with no code changes** — only the install differs. A CPU
> build of JAX comes in automatically. If you have a GPU and want to use it,
> add the right JAX build *after* setup, following
> [`roman_disperser/INSTALL.md` → GPU support](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#gpu-support).
> Don't pin JAX yourself — let the disperser pull the floor, then overlay GPU.

On **NERSC you don't install anything** — a shared environment is already built.
Only **laptop** users install the library themselves (and so are the only ones
who need GitHub access to the private repo).

---

## 1. NERSC

Curated shared environments already exist (`tutorial_2026_cpu` /
`tutorial_2026_gpu`); don't build your own. `.grism_sim_setup` exports their
paths and `ROMAN_DISPERSER_DATA` (the shared, read-only reference data).

**Activate (shell):**

```bash
source /global/common/software/m4943/.grism_sim_setup
module load conda
conda activate $tutorial_2026_cpu     # or $tutorial_2026_gpu on a GPU node
```

**Register the notebook kernels (once per user).** This installs both kernels,
names them from the active env, and wraps each in `kernel-helper.sh`, which
carries `ROMAN_DISPERSER_DATA` into notebooks — so data resolution is automatic
in shells *and* kernels:

```bash
source /global/common/software/m4943/.grism_sim_setup
module load conda
for V in cpu gpu; do
    envvar="tutorial_2026_$V"; conda activate "${!envvar}"
    KNAME=$(basename "$CONDA_PREFIX")                       # roman-tutorials-cpu / -gpu
    python -m ipykernel install --user --name "$KNAME" \
        --display-name "Roman Disperser Tutorials ($V)"
    sed -i '/"argv": \[/a\  "/global/common/software/m4943/kernel-helper.sh",' \
        "$HOME/.local/share/jupyter/kernels/$KNAME/kernel.json"
done
```

In the Jupyter launcher you'll see **Roman Disperser Tutorials (cpu)** and
**(gpu)** — use the GPU kernel only in a GPU-node session. The data is already
hydrated in the shared environment; nothing else to do. Go to §3.

---

## 2. Your own laptop (CPU)

This is the only path that installs the disperser from its **private** GitHub
repo, so set up GitHub auth first (org membership grants access, but you still
prove who you are). Two options — pick one and use the matching URL below:

- **SSH** (if your SSH key is registered with GitHub): nothing to set up; use the
  `git+ssh://git@github.com/...` form of the install URL.
- **HTTPS**: run `gh auth login`, which installs a git credential helper pip
  will use; use the `git+https://github.com/...` form.

Create the environment. The tutorials need only pip-installable packages
(`roman_disperser[full]` pulls jax, numpy, scipy, matplotlib, pandas, pyarrow,
zarr, astropy, synphot), so a **venv is the lightest path** — no conda required.

**Option A — venv + pip (recommended):**

Run this from the cloned tutorials repo; `.venv` (and `./data` below) are
git-ignored, so the venv, the data, and the notebooks all live in one folder —
the same layout the maintainer pixi setup uses.

```bash
python -m venv .venv          # in the repo root; .venv is git-ignored
source .venv/bin/activate
# SSH (recommended if your key is on GitHub):
pip install "roman_disperser[full] @ git+ssh://git@github.com/roman-grs-pit/roman_disperser.git@v0.10.0" \
    jupyterlab ipykernel
# — or HTTPS (after `gh auth login`):
#   git+https://github.com/roman-grs-pit/roman_disperser.git@v0.10.0
```

**Option B — conda** (use this if you'll also run the romanisim wrap later — it
needs conda for `fftw`; the current tutorials don't):

```bash
conda env create -f environment-cpu.yml      # no conda? `brew install micromamba` is the lightest
conda activate roman-disperser-tutorials
```

Then register a Jupyter kernel (either option):

```bash
python -m ipykernel install --user --name roman-tutorials \
    --display-name "Roman Disperser Tutorials"
```

**Hydrate the reference data.** The tutorials use **SCA 5**, so fetch just what
they need (a few hundred MB) rather than all 18 SCAs:

```bash
export ROMAN_DISPERSER_DATA=$PWD/data        # co-located in the repo (./data is git-ignored); or any stable path
roman-disperser-hydrate --only optical_model,sensitivities,synphot   # essentials (~2 MB)
roman-disperser-hydrate --only psf --sca 5                          # PSF cache for SCA 5
roman-disperser-hydrate --only catalog                              # source catalog (~155 MB; notebook 06)
# (or just `roman-disperser-hydrate` for everything — all 18 SCAs, ~4.5 GB)
```

**Make the data visible to the kernel.** A Jupyter kernel does **not** inherit
your shell, so add `ROMAN_DISPERSER_DATA` to the kernel's `kernel.json`. The
kernel's location is platform-dependent (macOS `~/Library/Jupyter`, Linux
`~/.local/share/jupyter`), so let Jupyter tell us where it is rather than
guessing — make sure `ROMAN_DISPERSER_DATA` is still exported, then run:

```bash
python - <<'PY'
import json, os
from jupyter_client.kernelspec import KernelSpecManager
p = os.path.join(KernelSpecManager().get_kernel_spec("roman-tutorials").resource_dir, "kernel.json")
d = json.load(open(p))
d.setdefault("env", {})["ROMAN_DISPERSER_DATA"] = os.environ["ROMAN_DISPERSER_DATA"]
json.dump(d, open(p, "w"), indent=2)
print("wired ROMAN_DISPERSER_DATA into", p)
PY
```

(`jupyter kernelspec list` shows the path too, if you'd rather edit
`kernel.json` by hand and add `"env": {"ROMAN_DISPERSER_DATA": "/your/path"}`.)
For GPU on an NVIDIA workstation, add the JAX overlay per the INSTALL link above
(Apple-silicon laptops are CPU-only).

Full hydration details (`--only`, manifests, lock files) are in
[`roman_disperser/INSTALL.md` → Reference data](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#reference-data).

---

## 3. Verify — run notebook 00

Launch JupyterLab from the repo (on NERSC, use [jupyter.nersc.gov](https://jupyter.nersc.gov)
instead of running it yourself):

```bash
jupyter lab          # opens in your browser
```

Select the **Roman Disperser Tutorials** kernel and run
**`notebooks/00_environment_check.ipynb`** top to bottom. It checks the import, the JAX backend, that the *kernel* resolves
the reference data, and a smoke dispersion — all with green ✅ markers. If
anything is red, fix it here (most often `ROMAN_DISPERSER_DATA` not reaching the
kernel, §2) before moving on.

---

## Developing the tutorials (maintainers)

Authoring uses [pixi](https://pixi.sh) via `pixi.toml` (not for users):

```bash
pixi shell             # laptop / CPU
pixi shell -e gpu      # linux GPU box
pixi run check-jax     # confirm the live backend
```

The disperser is private, so a local pixi solve needs GitHub auth too
(`gh auth login`, or switch the URL to `git+ssh://`). Notebooks are committed
**without outputs** via an `nbstripout` git filter — run `pixi run
setup-nbstripout` once per clone. Local runs catch gross breakage but **do not**
match the curated NERSC conda env; execute every notebook on NERSC in the real
environment before publishing. See [CLAUDE.md](../CLAUDE.md) for the rationale.
