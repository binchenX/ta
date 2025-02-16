# Ta

A command-line AI/LLM tool.

## Features (so far)

- Command line
- Nice Markdown display in terminal
- Choose model
- Thread Management
- RAG/Knowledge Base (local markdown files only)

## Help

```bash
$ ta
(4o) ❓>: /help
Commands:
/lm: List models
/lt: List threads
/st [thread_id]: Set current thread
/dt [thread_id]: Delete thread
/nt: New thread
/chat: Switch to chat mode
/chat model: Switch chat model to model
/rag: Switch to RAG mode
/rag index    : index new directory to knowledge base
/rag index -f : force index all directories to knowledge base
```

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
