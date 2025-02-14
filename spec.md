## Feature requirements

I want to write a command line software that:

- Asks questions and answers them using my local knowledge base saved in a path, most of which are markdown files.
- Saves all my conversations and follow-up questions, or keeps histories.
- I will involve this as an agent, e.g., ask it to do something automatically. RAG is the starting point, but don't limit the design and implementation to RAG. Make sure the chosen framework allows me to create an agent. This is very important.

## Design and implementation:

- It must be fully local, and the only external dependency will be the AI/LLM endpoint.
- All other dependencies needed must be local. It is okay to install those dependencies on the local machine. I'm fine with Docker but don't want to get into the trouble of debugging Docker network and port forwarding first. I want to get this running as quickly as I can.
- As mentioned above, make sure the chosen framework allows me to create an agent. This is very important.
- I prefer command line but a web UI may make sense. I also want to try some popular AI-focused UIs such as Streamlit (may be misspelled).

