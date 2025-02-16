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

## Config

see [example config](./config_example.toml).

## License

This project is licensed under the MIT [License](./LICENSE)
