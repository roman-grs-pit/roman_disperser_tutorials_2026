# Roman Disperser - Conda Environment

## Paths on NERSC

You can set all of these variable with `source /global/common/software/m4943/.grism_sim_setup`.

```bash
github_dir="/global/common/software/m4943/grizli0"
roman="/global/common/software/m4943/roman"
CRDS_SERVER_PATH="/global/cfs/cdirs/m4943/grismsim/crds_cache"
STPSF_PATH="/dvs_ro/cfs/cdirs/m4943/grismsim/stpsf-data"
```

## Quickstart

### For shell use

```
source /global/common/software/m4943/.grism_sim_setup
module load conda; conda activate $roman
```

### For notebooks - first time setup

```
source /global/common/software/m4943/.grism_sim_setup

module load conda; conda activate $roman

python -m ipykernel install \
    --user --name roman --display-name Roman-Disperser # Or whatever you'd prefer

cd $HOME/.local/share/jupyter/kernels/roman

sed -i '/"argv": \[/a\  "/global/common/software/m4943/kernel-helper.sh",' kernel.json
```
