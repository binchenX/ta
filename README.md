# Ta

A command-line AI/LLM tool to [scratch my own itch](spec.md).

## Features (so far)

- Command line
- Choose model
- Thread Managment
- RAG/Knowledge Base

```
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

see [example config](./config_example.toml)

## License

This project is licensed under the MIT [License](./LICENSE)
