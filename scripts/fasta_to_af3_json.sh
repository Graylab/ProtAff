#!/bin/bash
# Convert .fasta files from data/boltz2/adaptyv/fasta/ to AF3 JSON inputs in data/af3/inputs/
# Multi-chain fastas are concatenated with ":" so fastatojson produces a single JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/structures/boltz2/adaptyv/fasta"
OUTPUT_DIR="${PROJECT_DIR}/data/structures/af3/inputs"

mkdir -p "$OUTPUT_DIR"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

count=0
for fasta in "$INPUT_DIR"/*.fasta; do
    [ -f "$fasta" ] || continue
    name=$(basename "$fasta" .fasta)
    outfile="${OUTPUT_DIR}/${name}.json"
    if [ -f "$outfile" ]; then
        echo "Skipping (exists): $name"
        continue
    fi

    # Concatenate multi-chain sequences with ":" into a single-entry fasta
    concat_fasta="${TMPDIR}/${name}.fasta"
    python3 -c "
seqs = []
seq = []
with open('$fasta') as f:
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if seq:
                seqs.append(''.join(seq))
                seq = []
        else:
            seq.append(line)
    if seq:
        seqs.append(''.join(seq))
with open('$concat_fasta', 'w') as f:
    f.write('>${name}\n')
    f.write(':'.join(seqs) + '\n')
"

    # Run fastatojson from output dir so the JSON lands there
    (cd "$OUTPUT_DIR" && fastatojson -i "$concat_fasta")

    if [ -f "$outfile" ]; then
        echo "Converted: $name"
        count=$((count + 1))
    else
        echo "WARNING: expected output not found for $name"
    fi
done

echo "Done. Converted $count files to $OUTPUT_DIR"
