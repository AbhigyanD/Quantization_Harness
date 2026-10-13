#!/usr/bin/env bash
# Commit and push whatever is new under results/ with the given message.
# Needs GITHUB_TOKEN (Colab/Kaggle secret) with write access to the repo. The token is passed
# per-command via an auth header and is never written to git config or printed.
# Usage: scripts/push_results.sh "Save perplexity results for gptq"
set -eu
MSG="$1"
git add results
git diff --cached --quiet && { echo "nothing new in results/"; exit 0; }
git commit -q -m "$MSG"
AUTH=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 | tr -d '\n')
git -c http.extraheader="Authorization: Basic $AUTH" push -q origin HEAD:main
echo "pushed: $MSG"
