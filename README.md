# 🚀 Repr CLI

> A beautiful CLI that analyzes your code repositories and generates a compelling developer profile using AI.

[![PyPI version](https://img.shields.io/pypi/v/repr-cli.svg)](https://pypi.org/project/repr-cli/)
[![Python versions](https://img.shields.io/pypi/pyversions/repr-cli.svg)](https://pypi.org/project/repr-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🤖 **AI-Powered Analysis** — Deep code understanding, not just stats
- 🔒 **Zero Data Retention** — Your code is analyzed but never stored
- 🏠 **Local LLM Support** — Use Ollama for completely local processing
- 📊 **Deep Repository Analysis** — Architecture, frameworks, complexity metrics
- 🎨 **Beautiful UI** — Rich terminal interface with progress indicators
- 📝 **Markdown Output** — Clean, readable profiles you can share anywhere

## 🔐 Privacy Options

Choose your privacy level:

| Mode | Description | Data Flow |
|------|-------------|-----------|
| **Cloud (ZDR)** | Zero Data Retention with cloud LLMs | Code sent → analyzed → immediately discarded |
| **Local** | Completely local with Ollama | Code never leaves your machine |
| **Offline** | Basic stats only, no AI | No network required |

### Zero Data Retention (Default)

When using cloud analysis:
- ✅ Code is sent over encrypted connections (TLS)
- ✅ Analysis happens in ephemeral containers
- ✅ **No code is ever stored or logged**
- ✅ LLM providers configured for zero retention
- ✅ Results returned, data discarded immediately

### Local LLM Mode

For complete local control:
```bash
# Use Ollama (recommended)
repr analyze ~/code --local --model llama3.2

# Use any OpenAI-compatible local server
repr analyze ~/code --local --api-base http://localhost:11434/v1
```

## 📸 Demo

```bash
$ repr analyze ~/code

╭──────────────────────────────────────────────────────────────╮
│  🚀  Repr CLI v0.1.0                                         │
╰──────────────────────────────────────────────────────────────╯

Discovering repositories...
Found 12 repositories in 1 path(s)

                        Analyzing                               
╭────────────────────┬──────────┬──────────┬──────────┬─────────╮
│ Repository         │ Language │  Commits │      Age │  Status │
├────────────────────┼──────────┼──────────┼──────────┼─────────┤
│ ecommerce-api      │ Python   │      340 │   1.5 yr │ ✓ Done  │
│ react-dashboard    │ TypeScript│     156 │   8 mo   │ ✓ Done  │
│ ml-experiments     │ Python   │      89 │   4 mo   │ ✓ Done  │
╰────────────────────┴──────────┴──────────┴──────────┴─────────╯

✓ Profile saved: ~/.repr/profiles/2024-01-15.md
```

## 🚦 Quick Start

### Installation

```bash
pip install repr-cli
```

### For Local LLM (Optional)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2
```

### Usage

**Generate your profile (cloud, ZDR):**

```bash
# Analyze repos with cloud AI (zero data retention)
repr analyze ~/code
```

**Use local LLM:**

```bash
# Use Ollama locally
repr analyze ~/code --local

# Specify model
repr analyze ~/code --local --model codellama

# Use custom local endpoint
repr analyze ~/code --local --api-base http://localhost:1234/v1
```

**Offline mode (stats only):**

```bash
# No AI, no network - just code metrics
repr analyze ~/code --offline
```

**View your profile:**

```bash
repr view
repr view --raw > profile.md
```

## 🛠️ Commands

### `repr analyze`

Analyze repositories and generate a developer profile.

```bash
repr analyze <paths...> [OPTIONS]

Options:
  --local              Use local LLM (Ollama) instead of cloud
  --model NAME         Local model to use (default: llama3.2)
  --api-base URL       Custom local LLM API endpoint
  --offline            Stats only, no AI analysis
  --no-cache           Re-analyze all repositories
  --verbose, -V        Show detailed logs
```

### `repr view`

View your generated profile.

```bash
repr view [OPTIONS]

Options:
  --profile NAME  View a specific profile
  --raw, -r       Output plain markdown
```

### `repr login`

Authenticate for cloud analysis.

```bash
repr login
```

### `repr logout`

Clear authentication.

```bash
repr logout
```

### `repr push`

Upload profile to repr.dev.

```bash
repr push [--profile NAME]
```

### `repr profiles`

List all saved profiles.

```bash
repr profiles
```

## ⚙️ Configuration

Config stored in `~/.repr/config.json`:

```json
{
  "version": 1,
  "settings": {
    "default_paths": ["~/code"],
    "skip_patterns": ["node_modules", "venv", ".venv", "vendor", "__pycache__", ".git"]
  },
  "llm": {
    "extraction_model": "gpt-4o-mini",
    "synthesis_model": "gpt-4o",
    "local_api_url": "http://localhost:11434/v1",
    "local_api_key": "ollama"
  }
}
```

### Environment Variables

```bash
# Use local LLM by default
export REPR_LOCAL=true
export REPR_MODEL=codellama

# Custom Ollama endpoint
export OLLAMA_HOST=http://localhost:11434
```

## 🔍 What Gets Analyzed

The CLI performs deep analysis of your repositories:

**Code Metrics:**
- Lines of code, comments, complexity
- Function/class counts
- Average file and function sizes

**Architecture Detection:**
- Project type (web app, API, CLI, library, ML project)
- Architecture patterns (MVC, clean architecture, microservices)
- Framework detection (React, Django, FastAPI, etc.)

**Quality Indicators:**
- Test coverage and test frameworks
- Documentation presence and quality
- Docstring coverage

**Technical Stack:**
- Languages and percentages
- Dependencies and notable libraries
- API endpoints detection

## 📋 Requirements

- Python 3.10+
- Git
- For local mode: [Ollama](https://ollama.com/) or compatible LLM server

## 📁 Directory Structure

```
~/.repr/
├── config.json          # Settings
├── profiles/            # Generated profiles
│   ├── 2024-01-15.md
│   └── 2024-01-20.md
└── cache/              # Analysis cache
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](https://github.com/repr-app/cli/blob/main/CONTRIBUTING.md).

```bash
git clone https://github.com/repr-app/cli.git
cd cli
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT License - see [LICENSE](LICENSE).

## 🔗 Links

- [Website](https://repr.dev)
- [Documentation](https://repr.dev/docs)
- [GitHub](https://github.com/repr-app/cli)

## 💬 Support

- Email: [hello@repr.dev](mailto:hello@repr.dev)
- Discord: [discord.gg/repr](https://discord.gg/repr)

---

<p align="center">
  Made with ❤️ by the Repr team
</p>
