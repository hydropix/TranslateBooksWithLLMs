# Translation Quality Benchmark System

This document describes the benchmark system for testing LLM translation quality across multiple languages and models.

## Overview

The benchmark system allows you to:

- **Test translation quality** of Ollama, OpenAI-compatible, or OpenRouter models across 40+ languages
- **Evaluate translations** using OpenRouter LLMs (Claude, GPT-4, etc.)
- **Generate detailed reports** with scores and rankings
- **Publish results** to the GitHub wiki automatically

## Quick Start

```bash
# Run quick benchmark (19 representative languages)
python -m benchmark.cli run --openrouter-key YOUR_OPENROUTER_KEY

# Run quick benchmark against an OpenAI-compatible backend
python -m benchmark.cli run --provider openai --openai-endpoint http://localhost:8080/v1 -m your-model

# Run full benchmark (all 40+ languages)
python -m benchmark.cli run --full --openrouter-key YOUR_OPENROUTER_KEY

# Generate and publish wiki pages
python -m benchmark.cli wiki-publish
```

## Prerequisites

1. **Translation backend**: Ollama, OpenAI-compatible endpoint, OpenRouter, or Requesty
2. **OpenRouter API key** for translation evaluation (get one at [openrouter.ai](https://openrouter.ai))
   - For Requesty translation, get a key at [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys)

## CLI Commands

### `run` - Execute Benchmark

Runs translation tests and evaluates quality.

```bash
python -m benchmark.cli run [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-m, --models MODEL [MODEL ...]` | Specific provider models to test. If omitted, auto-detects available models from the selected provider |
| `-l, --languages CODE [CODE ...]` | Language codes to test (e.g., `fr de ja zh`). Default: quick test set (19 languages) |
| `--full` | Run full benchmark with all 40+ languages |
| `-p, --provider {ollama,openai,openrouter,requesty}` | Translation backend to benchmark |
| `--openai-key KEY` | API key for OpenAI-compatible translation backends |
| `--openai-endpoint URL` | OpenAI-compatible endpoint or `/v1` base URL |
| `--openrouter-key KEY` | OpenRouter API key (can also use `OPENROUTER_API_KEY` env var) |
| `--requesty-key KEY` | Requesty API key (can also use `REQUESTY_API_KEY` env var) |
| `--evaluator MODEL` | OpenRouter model for evaluation (default: `anthropic/claude-haiku-4.5`) |
| `--ollama-endpoint URL` | Custom Ollama endpoint (default: from `.env` or `http://localhost:11434/api/generate`) |
| `--resume RUN_ID` | Resume an interrupted benchmark run |

**Examples:**

```bash
# Test specific models on specific languages
python -m benchmark.cli run -m llama3:8b qwen2.5:14b mistral:7b -l fr de ja zh

# Test an OpenAI-compatible server
python -m benchmark.cli run -p openai --openai-endpoint http://localhost:8080/v1 -m qwen2.5-14b-instruct -l fr de ja zh

# Use a different evaluator model
python -m benchmark.cli run --evaluator anthropic/claude-3.5-sonnet --openrouter-key KEY

# Resume an interrupted run
python -m benchmark.cli run --resume abc12345 --openrouter-key KEY

# Custom Ollama endpoint
python -m benchmark.cli run --ollama-endpoint http://192.168.1.100:11434/api/generate
```

### `wiki` - Generate Wiki Pages

Generates markdown pages from benchmark results (local only).

```bash
python -m benchmark.cli wiki [RUN_ID]
```

- If `RUN_ID` is omitted, uses the latest completed run
- Output goes to `./wiki/` directory

### `wiki-publish` - Generate and Publish to GitHub

Generates wiki pages AND publishes them to the GitHub wiki repository.

```bash
python -m benchmark.cli wiki-publish [RUN_ID]
```

**Process:**
1. Generates wiki pages from benchmark results
2. Clones/updates the wiki repository
3. Copies generated files
4. Commits and pushes changes

**Requirements:**
- Git must be configured with push access to the wiki repo
- Wiki must be initialized on GitHub (create at least one page first)
- `WIKI_REPO_URL` env var or default `https://github.com/hydropix/TranslateBookWithLLM.wiki.git`

### `list` - List Benchmark Runs

Shows all available benchmark runs.

```bash
python -m benchmark.cli list
```

**Output:**
```
Run ID               Status       Started              Models                         Results
----------------------------------------------------------------------------------------------------
abc12345             completed    2025-01-15 10:30:00  llama3:8b, qwen2.5:14b (+1)   285
def67890             running      2025-01-15 14:22:00  mistral:7b                     42
```

### `show` - Show Run Details

Displays details of a specific benchmark run.

```bash
python -m benchmark.cli show RUN_ID [-d|--detailed]
```

**Options:**
- `-d, --detailed`: Show per-model and per-language statistics

### `export` - Export to CSV

Exports benchmark results to a CSV file.

```bash
python -m benchmark.cli export RUN_ID [-o OUTPUT_PATH]
```

**Example:**
```bash
python -m benchmark.cli export abc12345 -o results/benchmark_jan15.csv
```

### `merge` - Merge Multiple Runs

Combines results from multiple benchmark runs into one.

```bash
python -m benchmark.cli merge RUN_ID1 RUN_ID2 [RUN_ID...] [-o OUTPUT_ID] [--publish]
```

**Options:**
- `-o, --output`: Custom ID for the merged run
- `--publish`: Regenerate and publish wiki after merging

**Use case:** Run benchmarks on different model sets separately, then merge for a complete report.

```bash
# Merge three separate runs
python -m benchmark.cli merge run1 run2 run3 -o combined_benchmark --publish
```

### `delete` - Delete a Run

Removes a benchmark run from storage.

```bash
python -m benchmark.cli delete RUN_ID [-f|--force]
```

## How Testing Works

### Reference Texts

The benchmark uses 5 classic literature excerpts (~500 characters each):

| Text | Author | Year | Style |
|------|--------|------|-------|
| Pride and Prejudice | Jane Austen | 1813 | Prose with irony |
| The Picture of Dorian Gray | Oscar Wilde | 1890 | Sensory/aesthetic description |
| A Study in Scarlet | Arthur Conan Doyle | 1887 | Dialogue & character voice |
| Walden | Henry David Thoreau | 1854 | Philosophical nature writing |
| Moby-Dick | Herman Melville | 1851 | Archaic literary prose |

### Languages Tested

**Quick Benchmark (19 languages):**
- European: French, German, Spanish, Italian, Portuguese, Polish
- Asian: Chinese (Simplified), Chinese (Traditional), Japanese, Korean, Vietnamese, Thai
- South Asian: Hindi, Bengali, Tamil
- Cyrillic: Russian, Ukrainian
- Semitic: Arabic, Hebrew

**Full Benchmark (40+ languages):**
- All quick languages plus:
- European: Dutch, Swedish, Danish, Norwegian, Finnish, Greek, Romanian, Hungarian, Czech
- Asian: Indonesian, Malay, Filipino
- Cyrillic: Bulgarian, Serbian
- Classical: Latin, Ancient Greek, Sanskrit
- Minority: Welsh, Basque, Catalan, Galician, Irish, Scottish Gaelic, Icelandic, Maltese

### Evaluation Criteria

Each translation is scored on a 1-10 scale:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Meaning preservation - does the translation convey the original meaning? |
| **Fluency** | Natural expression - does it read naturally in the target language? |
| **Style** | Literary style preservation - is the author's voice maintained? |
| **Overall** | Global quality score combining all aspects |
| **Feedback** | Brief textual explanation of the scores |

### Score Indicators

| Score | Indicator | Label |
|-------|-----------|-------|
| 9-10 | 🟢 | Excellent |
| 7-8 | 🟡 | Good |
| 5-6 | 🟠 | Acceptable |
| 3-4 | 🔴 | Poor |
| 1-2 | ⚫ | Failed |

## Configuration

### Environment Variables

Set in `.env` file:

```bash
# Ollama Configuration
API_ENDPOINT=http://localhost:11434/api/generate
DEFAULT_MODEL=qwen3:14b
OLLAMA_NUM_CTX=2048
REQUEST_TIMEOUT=900

# OpenRouter Configuration (for evaluation)
OPENROUTER_API_KEY=your_key_here

# Requesty Configuration (OpenAI-compatible translation provider)
# Get a key at https://app.requesty.ai/api-keys
REQUESTY_API_KEY=your_key_here

# Wiki Publishing
WIKI_REPO_URL=https://github.com/your-username/TranslateBookWithLLM.wiki.git
```

### Default Evaluator

The default evaluator model is `anthropic/claude-haiku-4.5` (fast and economical).

Other recommended evaluators:
- `anthropic/claude-3.5-sonnet` - Higher quality, slower
- `openai/gpt-4o` - Alternative high-quality option
- `google/gemini-pro-1.5` - Google alternative

## Results Storage

Results are saved as JSON files in `./benchmark_results/`:

```
benchmark_results/
├── run_abc12345.json
├── run_def67890.json
└── ...
```

Each run file contains:
- Run metadata (ID, timestamps, status)
- List of models and languages tested
- All translation results with scores
- Error information for failed translations

## Wiki Output

Generated wiki pages include:

```
wiki/
├── Home.md              # Overview with top models and key stats
├── All-Languages.md     # Complete language ranking table
├── All-Models.md        # Complete model ranking table
├── languages/
│   ├── French.md        # Detailed results for French
│   ├── Japanese.md      # Detailed results for Japanese
│   └── ...
└── models/
    ├── llama3-8b.md     # Detailed results for llama3:8b
    ├── qwen2.5-14b.md   # Detailed results for qwen2.5:14b
    └── ...
```

## Testing New Models

### Step 1: Install the Model in Ollama

```bash
ollama pull your-new-model:tag
```

### Step 2: Run Benchmark on the New Model

```bash
# Test only the new model
python -m benchmark.cli run -m your-new-model:tag --openrouter-key KEY

# Or run alongside other models for comparison
python -m benchmark.cli run -m llama3:8b qwen2.5:14b your-new-model:tag --openrouter-key KEY
```

### Step 3: Review Results

```bash
# List runs to find your run ID
python -m benchmark.cli list

# Show detailed results
python -m benchmark.cli show RUN_ID --detailed

# Export to CSV for analysis
python -m benchmark.cli export RUN_ID -o new_model_results.csv
```

### Step 4: Merge and Publish (Optional)

If you want to add the new model to existing results:

```bash
# Merge with previous run
python -m benchmark.cli merge previous_run_id new_run_id -o complete_benchmark --publish
```

## Typical Workflow

### Initial Full Benchmark

```bash
# 1. Run full benchmark on all available models
python -m benchmark.cli run --full --openrouter-key KEY

# 2. Generate and publish wiki
python -m benchmark.cli wiki-publish

# 3. View results at your GitHub wiki
```

### Adding a New Model

```bash
# 1. Install the new model
ollama pull new-model:latest

# 2. Run quick benchmark to test
python -m benchmark.cli run -m new-model:latest --openrouter-key KEY

# 3. If results look good, run full benchmark
python -m benchmark.cli run -m new-model:latest --full --openrouter-key KEY

# 4. Merge with existing results
python -m benchmark.cli merge existing_run_id new_run_id --publish
```

### Interrupted Run Recovery

```bash
# If a run was interrupted, resume it
python -m benchmark.cli run --resume RUN_ID --openrouter-key KEY
```

## Architecture

```
CLI (benchmark/cli.py)
    │
    ▼
BenchmarkRunner (benchmark/runner.py)
    ├── Load Languages (from languages.yaml)
    ├── Load Reference Texts (from reference_texts.yaml)
    ├── BenchmarkTranslator (benchmark/translator.py)
    │       └── OllamaProvider (reuses main app's LLM client)
    ├── TranslationEvaluator (benchmark/evaluator.py)
    │       └── OpenRouter API (async HTTP)
    └── ResultsStorage (benchmark/results/storage.py)
            └── JSON files in benchmark_results/

WikiGenerator (benchmark/wiki/generator.py)
    ├── Load benchmark results
    ├── Render Jinja2 templates
    └── Generate markdown pages
```

## Troubleshooting

### "No Ollama models found"

Ensure Ollama is running and has at least one model:
```bash
ollama list
ollama pull llama3:8b
```

### "Failed to clone wiki repo"

1. Ensure you've created the wiki on GitHub (add at least one page)
2. Check your Git authentication
3. Verify the `WIKI_REPO_URL` is correct

### "OpenRouter API key not configured"

Either:
- Set `OPENROUTER_API_KEY` in your `.env` file
- Pass `--openrouter-key YOUR_KEY` to the command

### Slow benchmarks

- Use `--full` only when needed; quick benchmark (19 languages) is much faster
- Consider testing fewer models at a time
- Check your Ollama server performance

## See Also

- [Live Benchmark Results](https://github.com/hydropix/TranslateBookWithLLM/wiki) - Published wiki with current results
- [CLAUDE.md](../CLAUDE.md) - Main project documentation
- [OpenRouter Documentation](https://openrouter.ai/docs) - API documentation for evaluation
