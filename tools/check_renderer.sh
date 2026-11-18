#!/bin/sh
# The renderer interface's check (multi_target_optimizer.h PreviewRenderer): converts a scan twice, without and with
# --check_renderer, and passes when the two pictures are identical byte for byte and no pass of the bundle search, the
# curve fit or the tone plan ran outside the renderer (neg_process exits 1 with --check_renderer when one does). Run it
# on a narrow-band (TriLED) scan and on a white-light scan with the curve fit, before a change to the search, the curve
# fit, the tone plan or the renderer interface (multi_target_optimizer.h) is merged, e.g.
#
#     tools/check_renderer.sh bin_out/neg_process --film_stock portra400 --light_source tri --roi auto -P srgb scan.ARW
#     tools/check_renderer.sh bin_out/neg_process --film_stock portra400 --light_source d65 --roi auto -P srgb scan.ARW
#
# The arguments after the binary are neg_process's, the scan last, without -o (the script writes into a temporary
# folder and puts -o before the scan). -P srgb runs the curve fit (white light) and the tone plan.
set -eu
if [ $# -lt 2 ]; then
  echo "usage: tools/check_renderer.sh <neg_process> <neg_process arguments, the scan last, without -o>" >&2
  exit 2
fi
np=$1
shift
# "$@" without its last argument, which is the scan.
for a do
  shift
  if [ -n "${scan+x}" ]; then set -- "$@" "$scan"; fi
  scan=$a
done
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if ! "$np" "$@" -o "$tmp/plain.tif" "$scan" > "$tmp/plain.log" 2>&1; then
  cat "$tmp/plain.log"
  echo "FAILED: the conversion without --check_renderer exited with an error"
  exit 1
fi
if ! "$np" "$@" --check_renderer -o "$tmp/checked.tif" "$scan" > "$tmp/checked.log" 2>&1; then
  grep -i -e "renderer check" -e "error" "$tmp/checked.log" || cat "$tmp/checked.log"
  echo "FAILED: the conversion with --check_renderer exited with an error"
  exit 1
fi
if ! grep "Renderer check" "$tmp/checked.log"; then
  echo "FAILED: the run with --check_renderer printed no renderer check (no bundle search ran?)"
  cat "$tmp/checked.log"
  exit 1
fi
if cmp -s "$tmp/plain.tif" "$tmp/checked.tif"; then
  echo "identical: the picture through the pass-through renderer equals the plain one"
else
  echo "DIFFERENT: the picture through the pass-through renderer differs from the plain one"
  exit 1
fi
