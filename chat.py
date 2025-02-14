import openai
import json
import os
from logger_config import configure_logging
from typing import Dict, List
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

logger = configure_logging()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Disable ChromaDB telemetry
chroma_settings = Settings(anonymized_telemetry=False, allow_reset=True)


class ChatOpenAI:
    def __init__(self, history_limit=5, save_file="conversations.json"):
        self.threads: Dict[str, Dict] = {}
        self.history_limit = history_limit
        self.save_file = save_file

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db", settings=chroma_settings
        )

        # Use OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBEDDING_NAME", "text-embedding-3-large_v1"),
        )

        # Create or get the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="thread_summaries", embedding_function=self.embedding_function
        )

        self.load_conversations()

    def load_conversations(self):
        """Load previous conversation history from a file."""
        if os.path.exists(self.save_file):
            with open(self.save_file, "r") as f:
                self.threads = json.load(f)

    def save_conversations(self):
        """Save conversation history to a file."""
        with open(self.save_file, "w") as f:
            json.dump(self.threads, f, indent=4)

    # delete thread by id
    def delete_thread(self, thread_id: str):
        """Delete a thread by ID."""
        if thread_id in self.threads:
            del self.threads[thread_id]
            self.save_conversations()
            return True

        # update the vector db
        try:
            self.collection.delete(ids=[thread_id])
        except Exception as e:
            logger.error(f"Error deleting thread from vector database: {e}")
        return False

    def list_threads_with_topics(self) -> List[str]:
        """List all thread IDs with their topics in the format [thread_id] topic."""
        return [
            f"[{thread_id}] {self.threads[thread_id].get('topic', 'No topic')}"
            for thread_id in self.threads
        ]

    # set current thread id
    def set_current_thread_id(self, thread_id: str):
        self.current_thread_id = thread_id

    def generate_thread_id(self) -> str:
        """Generate a unique thread ID."""
        if not self.threads:
            return "1"
        if self.threads:
            last_thread_id = max(int(tid) for tid in self.threads.keys())
            return str(last_thread_id + 1)

    def generate_topic_and_summary(self, messages: List[Dict]) -> tuple:
        """Generate a topic and summary for a thread using GPT."""
        context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        prompt = f"""Based on the following conversation, provide:
        1. A topic (less than 10 words)
        2. A summary (up to 100 words)
        
        Conversation:
        {context}
        
        Format your response as:
        TOPIC: [topic]
        SUMMARY: [summary]"""

        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4-mini_v2024-07-18"),
            messages=[{"role": "user", "content": prompt}],
        )

        result = response.choices[0].message.content
        topic = result.split("TOPIC:")[1].split("SUMMARY:")[0].strip()
        summary = result.split("SUMMARY:")[1].strip()

        logger.info(f"Generated topic: {topic}")

        return topic, summary

    def update_vector_db(self, thread_id: str, summary: str):
        """Update the vector database with the thread summary."""
        try:
            # Check if thread_id exists in the collection
            results = self.collection.get(ids=[thread_id], include=["documents"])

            # log thread_id and summary
            logger.info(f"Update vector db: Thread ID: {thread_id}, Topic: {summary}")

            if results and results["ids"]:
                # Update existing entry
                self.collection.update(ids=[thread_id], documents=[summary])
            else:
                # Add new entry
                self.collection.add(ids=[thread_id], documents=[summary])
        except Exception as e:
            logger.error(f"Error updating vector database: {e}")

    # don't use for now, use simple thread management
    # each time start the program will create a new thread use uuid
    def detect_thread(self, user_message: str) -> str:
        """Find an existing thread with a similar topic using vector similarity search."""
        try:
            # Query the vector database
            results = self.collection.query(
                query_texts=[user_message],
                n_results=1,
                include=["distances", "documents"],
            )

            if results and results["distances"] and results["distances"][0]:
                similarity_score = (
                    1 - results["distances"][0][0]
                )  # Convert distance to similarity
                logger.info(f"Similarity score: {similarity_score}")
                if similarity_score > 0.7:  # Threshold
                    return results["ids"][0][0]

            logger.info("No similar thread found")
            # Create new thread if no similar thread found
            return str(len(self.threads) + 1)

        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return str(len(self.threads) + 1)

    def add_message(self, thread_id: str, role: str, message: str):
        """Add a message to the conversation thread and update metadata."""
        if thread_id not in self.threads:
            self.threads[thread_id] = {"messages": [], "topic": "", "summary": ""}

        self.threads[thread_id]["messages"].append({"role": role, "content": message})

        # Keep only last few messages
        self.threads[thread_id]["messages"] = self.threads[thread_id]["messages"][
            -self.history_limit :
        ]
        self.save_conversations()

    def add_turn(self, thread_id: str, usr_message: str, asssitant_message: str):
        self.add_message(thread_id, "user", usr_message)
        self.add_message(thread_id, "assistant", asssitant_message)

        # update topic and summary with lastest exchange, and update the vector db with thread topic
        topic, summary = self.generate_topic_and_summary(
            self.threads[thread_id]["messages"]
        )
        self.threads[thread_id]["topic"] = topic
        self.threads[thread_id]["summary"] = summary
        self.update_vector_db(thread_id, topic)

        # save topic and sumary
        self.save_conversations()

    def query(self, user_message: str) -> str:
        """Process user message and get AI response while managing conversation history."""
        thread_id = self.current_thread_id

        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        messages += self.get_recent_messages(thread_id)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"), messages=messages
        )

        assistant_reply = response.choices[0].message.content

        self.add_turn(thread_id, user_message, assistant_reply)

        logger.info(
            f"Thread ID: {thread_id}, Topic: {self.threads[thread_id].get('topic', 'No topic yet')}"
        )

        return assistant_reply

    def get_recent_messages(self, thread_id: str) -> List[Dict]:
        """Retrieve recent messages from a thread for context."""
        return self.threads.get(thread_id, {}).get("messages", [])

    def get_thread_info(self, thread_id: str) -> Dict:
        """Get thread information including topic and summary."""
        if thread_id in self.threads:
            return {
                "topic": self.threads[thread_id].get("topic", ""),
                "summary": self.threads[thread_id].get("summary", ""),
                "message_count": len(self.threads[thread_id]["messages"]),
            }
        return None
