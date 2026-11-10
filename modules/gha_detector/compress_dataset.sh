#!/usr/bin/env bash

set -euo pipefail

input_file="gha_smells.csv"
output_file="${input_file}.xz"
hash_file="${input_file}.sha256.txt"

if [[ ! -f "${input_file}" ]]; then
	echo "Input file not found: ${input_file}" >&2
	exit 1
fi

if ! command -v xz >/dev/null 2>&1; then
	echo "xz is not installed or not available in PATH" >&2
	exit 1
fi

if ! command -v sha256sum >/dev/null 2>&1; then
	echo "sha256sum is not installed or not available in PATH" >&2
	exit 1
fi

xz -9e -T0 -k -f "${input_file}"
sha256sum "${input_file}" > "${hash_file}"

echo "Compressed ${input_file} -> ${output_file}"
echo "Saved hash to ${hash_file}"
