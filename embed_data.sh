#!/bin/sh
# Writes an assembly file that embeds the given data files in neg_process (.incbin) with a table of
# {name, data, size} entries read by embedded_data.cc. Each file is followed by a NUL byte, so text files can be
# parsed in place. Paths are embedded as given (relative to the repository root, where make runs).
#
#   sh embed_data.sh profiles/profile_X.json models/dinov3-small_portra400/adapter.onnx ... > bin_out/embedded_data.S
set -e

i=0
echo '    .section .rodata'
for f in "$@"; do
  case "$f" in
    *'"'*|*'\'*) echo "embed_data.sh: unsupported file name: $f" >&2; exit 1 ;;
  esac
  [ -f "$f" ] || { echo "embed_data.sh: missing file: $f" >&2; exit 1; }
  echo '    .balign 16'
  echo "negicc_embedded_data_$i:"
  echo "    .incbin \"$f\""
  echo "negicc_embedded_data_${i}_end:"
  echo '    .byte 0'
  echo "negicc_embedded_name_$i:"
  echo "    .asciz \"$f\""
  i=$((i + 1))
done

echo '    .section .data.rel.ro, "aw"'
echo '    .balign 8'
echo '    .globl negicc_embedded_table'
echo 'negicc_embedded_table:'
i=0
for f in "$@"; do
  echo "    .quad negicc_embedded_name_$i, negicc_embedded_data_$i, negicc_embedded_data_${i}_end - negicc_embedded_data_$i"
  i=$((i + 1))
done
echo '    .globl negicc_embedded_count'
echo 'negicc_embedded_count:'
echo "    .quad $i"
echo '    .section .note.GNU-stack, "", %progbits'
