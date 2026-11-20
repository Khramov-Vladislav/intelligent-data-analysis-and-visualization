#!/bin/sh
set -eu

dv_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
typst_dir="$dv_dir/typst"

for source in "$dv_dir"/src/*.md; do
    name=${source##*/}
    pandoc "$source" \
        --from=gfm+implicit_figures-strikeout \
        --to=typst --standalone --wrap=none \
        --shift-heading-level-by=-1 \
        --lua-filter="$typst_dir/keep-image-labels.lua" \
        --template="$typst_dir/report.template" \
        --output="$typst_dir/${name%.md}.typ"
done
