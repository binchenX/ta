import sys

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.markdown import Markdown

from llm import LLM
from log import configure_logging

logger = configure_logging()


console = Console()


class HackerNews:
    BEST_PAGE_URL = "https://news.ycombinator.com/best"
    ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"

    def __init__(self):
        self.model = "gpt-4o-mini_v2024-07-18"
        self.llm = LLM(self.model)

    def get_best_stories(self, limit=20):
        """Fetch the best stories by scraping the 'Best' page."""
        try:
            # Fetch the HTML from the Best page
            response = requests.get(self.BEST_PAGE_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract story IDs from <tr class="athing"> elements
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

    def summarize_text(self, text):
        """Summarize text using the LLM wrapper."""
        return self.llm.summarize(text)

    def summarize_stories(self, stories):
        """Return stories as markdown formatted strings."""
        markdown_summaries = []
        for story in stories:
            title = story.get("title", "No title")
            url = story.get("url", "No URL")  # Default to "No URL" if not present
            text = f"## {title}\nURL: {url}"

            if "text" in story:
                text += f"\n\nContent: {story['text']}"

            logger.debug(f"Summarizing story: {title}")
            summary = self.summarize_text(text)
            # Format as markdown string
            markdown_story = f"## [{title}]({url})\n>{summary}"
            markdown_summaries.append(markdown_story)

        return markdown_summaries

    def display_summaries(self, summaries):
        """Display markdown formatted summaries."""
        for i, news in enumerate(summaries, start=1):
            console.print(Markdown(news))

    def do(self, limit=3):
        best_stories = self.get_best_stories(limit=limit)
        summaries = self.summarize_stories(best_stories)
        self.display_summaries(summaries)


def main():
    if len(sys.argv) != 1:
        print("Usage: python hacker_news.py")
        sys.exit(1)

    HackerNews().do(limit=3)


if __name__ == "__main__":
    main()
