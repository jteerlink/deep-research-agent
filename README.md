# Standalone Ollama Exa Leadgen

This branch intentionally contains only the lightweight lead-generation surfaces:

- `.agents/skills/lead-research-assistant/SKILL.md` — prompt skill for qualitative lead research and outreach strategy.
- `standalone/ollama_exa_leadgen/` — isolated CLI for Ollama Cloud planning plus Exa discovery.
- `docs/standalone-ollama-exa-leadgen-spec.md` — implementation spec and rationale.

The original deep research agent is preserved on the `archive/deep-research-agent` branch.

## Quick start

```bash
cd standalone/ollama_exa_leadgen
PYTHONPATH=src python3 -m ollama_exa_leadgen plan --icp-file examples/dental-dfw-icp.json --target-count 5
```

Live runs require `EXA_API_KEY` and `OLLAMA_API_KEY`.
