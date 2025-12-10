#!/usr/bin/env bash
# A handful of Laravel docs pages — small enough to index in a minute.
set -euo pipefail
mkdir -p corpus
for page in eloquent eloquent-relationships queues routing validation cache scheduling middleware; do
  curl -fsSL "https://raw.githubusercontent.com/laravel/docs/12.x/$page.md" -o "corpus/$page.md"
done
ls corpus
