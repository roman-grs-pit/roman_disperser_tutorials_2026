#!/usr/bin/env bash
# One-time per clone: configure the nbstripout git filter so notebook outputs
# are stripped in what git stores (commits stay clean) while your working-copy
# outputs are left intact. The filter shells out via `pixi run -e dev` so it
# works from any shell, with or without the pixi env active.
#
#   pixi run setup-nbstripout      # or: bash scripts/setup_nbstripout.sh
#
# Git config lives in .git/config (not version-controlled), so each fresh clone
# runs this once. .gitattributes (which maps *.ipynb -> this filter) IS
# committed. To undo: `git config --remove-section filter.nbstripout`.
set -euo pipefail
cd "$(dirname "$0")/.."

git config filter.nbstripout.clean "pixi run -e dev nbstripout"
git config filter.nbstripout.smudge "cat"
git config filter.nbstripout.required true
git config diff.ipynb.textconv "pixi run -e dev nbstripout -t"

echo "nbstripout git filter configured for this clone."
echo "Committed notebooks will have outputs stripped automatically."
