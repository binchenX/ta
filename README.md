# Ta

A command-line AI/LLM tool.

[![asciicast](https://asciinema.org/a/XEQS8Qx3sBH7yDeEEHkADODeO.svg)](https://asciinema.org/a/XEQS8Qx3sBH7yDeEEHkADODeO)

## Features (so far)

- Command line
- Nice Markdown display in terminal
- Choose model
- Thread Management
- RAG/Knowledge Base (local markdown files only)
- Tool Use (local file, mcp, internet, brave search)
- Agent (kind of) deepsearch using brave search


## Install

As usual.

## Config

1. First, set the environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_BASE_URL=your_openai_base_url
```

2. Configure the `model_aliases`, `chat`, and `rag` sections. See the [example
config](./config_example.toml). Place the config in `~/.td/config.toml`.

## License

This project is licensed under the MIT [License](./LICENSE)
