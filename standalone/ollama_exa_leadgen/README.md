# Ollama Exa Leadgen

Standalone lead-generation CLI that uses **Ollama Cloud only** for LLM planning/QA and **Exa Deep Search** for web-grounded prospect discovery.

This subproject is intentionally isolated from the root `deep_research_agent` package: no imports, no shared runtime modules, and no root dependency changes.

## Quick start

```bash
cd standalone/ollama_exa_leadgen
cp examples/dental-dfw-icp.json /tmp/icp.json
PYTHONPATH=src python3 -m ollama_exa_leadgen plan --icp-file /tmp/icp.json
```

Live run requires:

```bash
export EXA_API_KEY=...
export OLLAMA_API_KEY=...
export OLLAMA_MODEL=deepseek-v4-pro:cloud
PYTHONPATH=src python3 -m ollama_exa_leadgen run --icp-file examples/dental-dfw-icp.json --target-count 10
```

Outputs are written under `runs/<campaign>/<timestamp>/` by default.
