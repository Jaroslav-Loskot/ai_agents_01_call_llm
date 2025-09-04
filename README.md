# LLM + Tool Orchestration Demo

This is a small Python project that demonstrates how to:

1. Call an **LLM API** (OpenAI).
2. Let the LLM decide whether to use a **math tool**.
3. Run the tool safely in Python (`safe_eval`).
4. Send the tool’s result **back to the LLM** for a clear final answer.

---

## 🚀 Features

* Uses [`uv`](https://github.com/astral-sh/uv) for fast dependency management.
* Secrets handled via `.env` (never commit your API key).
* Simple **agent trace mode** (`--trace`) to see each step.
* Safe arithmetic evaluation:

  * `+ - * / // % **`
  * Parentheses
  * `sqrt(x)` (square root)
* Colorized CLI output.
* ✅ Ready-to-use **Docker** and **Docker Compose** setup.

---

## 📦 Setup with `uv`

### 1. Install `uv`

Windows (PowerShell):

```powershell
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
```

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close & reopen your terminal, then verify:

```bash
uv --version
```

---

### 2. Clone & prepare project

```bash
git clone https://github.com/Jaroslav-Loskot/ai_agents_01_call_llm
cd ai_agents_01_call_llm
```

---

### 3. Configure environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
```

---

### 4. Create environment & install dependencies

```bash
uv venv .venv
uv add openai python-dotenv
```

---

## 🖥 Usage (local)

### Run with default question

```bash
uv run python src/ai_agents_01_call_llm/main.py
```

### Custom question

```bash
uv run python src/ai_agents_01_call_llm/main.py "What is sqrt(144) + 5 ?"
```

### Show detailed trace

```bash
uv run python src/ai_agents_01_call_llm/main.py "What is 2**10 + sqrt(81) ?" --trace
```

---

## 🐳 Docker Usage

### 1. Build image

```bash
docker build -t ai-agents-llm .
```

### 2. Run with inline query

```bash
docker run --rm -it \
  --env-file .env \
  ai-agents-llm "What is sqrt(256) + 10 ?" --trace
```

This uses the `ENTRYPOINT` defined in the Dockerfile.

---

## 📦 Docker Compose

A `docker-compose.yml` is included for convenience:

```yaml
version: "3.9"

services:
  llm-tool:
    build: .
    image: ai-agents-llm:latest
    container_name: llm-tool
    env_file:
      - .env
    stdin_open: true
    tty: true
    command: ["What is sqrt(256) + 10 ?", "--trace"]
```

### Run with default query

```bash
docker compose run --rm llm-tool
```

### Run with custom query

```bash
docker compose run --rm llm-tool "What is 2**10 ?" --trace
```

---

## 📂 Project Structure

```
.
├── src/
│   └── ai_agents_01_call_llm/
│       └── main.py        # Main orchestration script
├── pyproject.toml         # Project metadata (uv)
├── uv.lock                # Locked dependency versions
├── Dockerfile             # Container definition
├── docker-compose.yml     # Compose service definition
├── .env.example           # Example environment file
├── .gitignore             # Ignore venv and secrets
└── README.md              # This file
```

---

## 🛡 Safety

The evaluator explicitly rejects variables, imports, or function calls other than:

* pure numeric arithmetic (`+ - * / // % **`)
* parentheses
* `sqrt(x)`

This prevents code injection or access to Python objects.

---

## 📝 License

MIT – feel free to use and modify.
