import logging
import sys
from typing import List

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.markdown import Markdown

from llm import LLM
from log import configure_logging

# Configure root logger and set hackernews to DEBUG
configure_logging(default_level=logging.INFO)
logger = logging.getLogger(__name__)  # "hackernews"
logger.setLevel(logging.DEBUG)

console = Console()


class Story:
    """Class representing a summarized news story."""

    def __init__(self, title: str, url: str, summary: str):
        self.title = title
        self.url = url
        self.summary = summary

    def to_markdown(self) -> str:
        """Convert the story to a Markdown string."""
        return f"## [{self.title}]({self.url})\n>{self.summary}"


class HackerNews:
    BEST_PAGE_URL = "https://news.ycombinator.com/best"
    ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"

    def __init__(self):
        self.model = "gpt-4o-mini_v2024-07-18"
        self.llm = LLM(self.model)

    def get_best_stories(self, limit=20) -> List[dict]:
        """Fetch the best stories by scraping the 'Best' page."""
        try:
            response = requests.get(self.BEST_PAGE_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            story_elements = soup.find_all("tr", class_="athing")[:limit]
            best_story_ids = [elem["id"] for elem in story_elements]

            best_stories = []
            for story_id in best_story_ids:
                story_response = requests.get(self.ITEM_URL.format(story_id))
                story_response.raise_for_status()
                story_details = story_response.json()
                best_stories.append(story_details)
                logger.debug(f"Fetched story: {story_details.get('title', 'No title')}")

            return best_stories

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching best stories: {e}")
            return []

    def summarize_text(self, text: str) -> str:
        """Summarize text using the LLM wrapper."""
        # Truncate to avoid overwhelming the LLM (adjust as needed)
        if len(text) > 10000:
            text = text[:10000]
            logger.debug("Truncated input text to 10,000 chars")
        return self.llm.summarize(text)

    def summarize_stories(self, stories: List[dict]) -> List[Story]:
        """Return stories as a list of Story objects with summaries from URL content."""
        story_objects = []
        for story in stories:
            title = story.get("title", "No title")
            url = story.get("url", "No URL")
            logger.debug(f"Processing story: {title}")

            if url == "No URL":
                summary = "No URL provided to summarize."
            else:
                try:
                    # Fetch URL content
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Try to find article-like content
                    article = soup.find("article") or soup.find(
                        "div", class_=["content", "post", "article"]
                    )
                    if article:
                        content = " ".join(
                            p.text.strip() for p in article.find_all("p") if p.text.strip()
                        )
                    else:
                        # Fallback to all <p> tags, excluding common noise classes
                        paragraphs = [
                            p for p in soup.find_all("p") if "footer" not in p.get("class", [])
                        ]
                        content = " ".join(p.text.strip() for p in paragraphs if p.text.strip())

                    if not content or len(content) < 100:
                        content = soup.get_text(strip=True)
                        logger.debug(
                            "Using full text fallback due to insufficient paragraph content"
                        )

                    logger.debug(f"Fetched {len(content)} chars from {url}")
                    if len(content) < 50:
                        summary = f"Content too short to summarize: {content[:50]}..."
                    else:
                        summary = self.summarize_text(content)
                except (requests.RequestException, Exception) as e:
                    logger.error(f"Error fetching {url}: {e}")
                    summary = f"Failed to fetch content: {str(e)}"

            story_objects.append(Story(title, url, summary))

        return story_objects

    def display_summaries(self, summaries: List[Story]) -> None:
        """Display summaries as Markdown."""
        for i, story in enumerate(summaries, start=1):
            console.print(Markdown(story.to_markdown()))

    def do(self, limit: int = 3) -> None:
        """Execute the main workflow."""
        best_stories = self.get_best_stories(limit=limit)
        summaries = self.summarize_stories(best_stories)
        self.display_summaries(summaries)


def main() -> None:
    if len(sys.argv) != 1:
        print("Usage: python hacker_news.py")
        sys.exit(1)

    HackerNews().do(limit=3)


if __name__ == "__main__":
    main()
