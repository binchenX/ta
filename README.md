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

This project is licensed under the MIT License.

Copyright (c) 2024 pierr.chen@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
