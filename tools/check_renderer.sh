#!/bin/sh
# The renderer interface's check (multi_target_optimizer.h PreviewRenderer): converts a scan twice, without and with
# --check_renderer, and passes when the two pictures are identical byte for byte and no pass of the bundle search, the
# curve fit or the tone plan ran outside the renderer. Run it on a narrow-band (TriLED) scan and on a white-light scan
# with the curve fit, e.g.
#
#     tools/check_renderer.sh bin_out/neg_process --film_stock portra400 --light_source tri --roi auto scan.ARW
#     tools/check_renderer.sh bin_out/neg_process --film_stock portra400 --light_source d65 --roi auto scan.ARW
#
# The arguments after the binary are neg_process's, without -o (the script writes into a temporary folder).
set -eu
np=$1
shift
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$np" "$@" -o "$tmp/plain.tif" > "$tmp/plain.log" 2>&1 || { cat "$tmp/plain.log"; exit 1; }
if ! "$np" "$@" --check_renderer -o "$tmp/checked.tif" > "$tmp/checked.log" 2>&1; then
  grep -i "renderer check\|error" "$tmp/checked.log"
  exit 1
fi
grep "Renderer check" "$tmp/checked.log"
if cmp -s "$tmp/plain.tif" "$tmp/checked.tif"; then
  echo "identical: the picture through the pass-through renderer equals the plain one"
else
  echo "DIFFERENT: the picture through the pass-through renderer differs from the plain one"
  exit 1
fi
