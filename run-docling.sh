#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
uv run -m docling.cli.main "$@"
