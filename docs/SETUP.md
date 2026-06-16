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
repo, so authenticate first (org membership grants access, but you still prove
who you are):

```bash
gh auth login          # sets up a git credential helper pip will use
```

Create the environment and register a kernel:

```bash
conda env create -f environment-cpu.yml
conda activate roman-disperser-tutorials
KNAME=$(basename "$CONDA_PREFIX")
python -m ipykernel install --user --name "$KNAME" \
    --display-name "Roman Disperser Tutorials"
```

**Hydrate the reference data.** The tutorials use **SCA 5**, so fetch just what
they need (a few hundred MB) rather than all 18 SCAs:

```bash
export ROMAN_DISPERSER_DATA=~/roman_disperser_data        # any stable path; add to ~/.bashrc
roman-disperser-hydrate --only optical_model,sensitivities,synphot   # essentials (~2 MB)
roman-disperser-hydrate --only psf --sca 5                          # PSF cache for SCA 5
roman-disperser-hydrate --only catalog                              # source catalog (~155 MB; notebook 06)
# (or just `roman-disperser-hydrate` for everything — all 18 SCAs, ~4.5 GB)
```

**Make the data visible to the kernel.** A Jupyter kernel does **not** inherit
your shell, so add `ROMAN_DISPERSER_DATA` to the kernel's `kernel.json`:

```bash
KJ=~/.local/share/jupyter/kernels/$(basename "$CONDA_PREFIX")/kernel.json
python - "$KJ" <<'PY'
import json, os, sys
p = sys.argv[1]; d = json.load(open(p))
d.setdefault("env", {})["ROMAN_DISPERSER_DATA"] = os.environ["ROMAN_DISPERSER_DATA"]
json.dump(d, open(p, "w"), indent=2)
PY
```

(Equivalently, edit `kernel.json` and add `"env": {"ROMAN_DISPERSER_DATA":
"/your/path"}`.) For GPU on an NVIDIA workstation, add the JAX overlay per the
INSTALL link above (Apple-silicon laptops are CPU-only).

Full hydration details (`--only`, manifests, lock files) are in
[`roman_disperser/INSTALL.md` → Reference data](https://github.com/roman-grs-pit/roman_disperser/blob/main/INSTALL.md#reference-data).

---

## 3. Verify — run notebook 00

Launch JupyterLab (or open the notebooks in NERSC Jupyter), select the **Roman
Disperser Tutorials** kernel, and run **`notebooks/00_environment_check.ipynb`**
top to bottom. It checks the import, the JAX backend, that the *kernel* resolves
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
