# Local Ollama Setup (gemma4:e2b)

This project is configured to run on a local Ollama model instead of OpenAI.

## Prerequisites

1. Install Ollama and ensure it is available locally.
2. Start the Ollama server:

```powershell
ollama serve
```

3. Ensure the model is available:

```powershell
ollama pull gemma4:e2b
ollama list
```

## Environment Variables

Create a `.env` file in the project root if needed.

```env
# Optional: defaults to gemma4:e2b
OLLAMA_MODEL=gemma4:e2b

# Optional: defaults to local Ollama server
OLLAMA_BASE_URL=http://localhost:11434

# Optional: only needed for Tavily search in stage_02_tools/02_web_search.py
TAVILY_API_KEY=tvly-...
```

`OPENAI_API_KEY` is not required for the current runnable scripts.

## Install Dependencies

```powershell
uv sync
```

## Run Tutorials

### Stage 1 Basics

```powershell
uv run .\deep_research\deep_research_agent\stage_01_basics\01_hello_langchain.py
uv run .\deep_research\deep_research_agent\stage_01_basics\02_prompt_templates.py
uv run .\deep_research\deep_research_agent\stage_01_basics\03_chains.py
```

### Stage 2 Tools

```powershell
uv run .\deep_research\deep_research_agent\stage_02_tools\01_custom_tools.py
uv run .\deep_research\deep_research_agent\stage_02_tools\02_web_search.py
```

## Notes

- `stage_02_tools/02_web_search.py` works without Tavily by falling back to DuckDuckGo.
- If you change local models later, update `OLLAMA_MODEL` only; code changes are not required.
