import asyncio
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
logger.setLevel(logging.CRITICAL)
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

    async def _playwright_get(self, url: str) -> str:
        """Fetch content using Playwright for JavaScript-rendered pages."""
        from playwright.async_api import async_playwright

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                # Wait a bit for dynamic content
                await page.wait_for_timeout(2000)
                # Get the text content
                text = await page.evaluate("() => document.body.innerText")
                await browser.close()
                return text
        except Exception as e:
            logger.error(f"Playwright error for {url}: {e}")
            return ""

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
                    # Regular fetch attempt first
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
                        paragraphs = [
                            p for p in soup.find_all("p") if "footer" not in p.get("class", [])
                        ]
                        content = " ".join(p.text.strip() for p in paragraphs if p.text.strip())

                    # If content is too short, try Playwright
                    if len(content) < 50:
                        logger.debug(f"Content too short, trying Playwright for {url}")
                        content = asyncio.run(self._playwright_get(url))
                        if not content:
                            summary = "Failed to fetch content with Playwright"
                            story_objects.append(Story(title, url, summary))
                            continue

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

    def _display_summaries(self, summaries: List[Story]) -> None:
        """Display summaries as Markdown."""
        for i, story in enumerate(summaries, start=1):
            console.print(Markdown(story.to_markdown()))

    def show(self, limit: int = 3) -> None:
        """Fetch and display the best stories."""
        best_stories = self.get_best_stories(limit=limit)
        summarized_stories = self.summarize_stories(best_stories)
        self._display_summaries(summarized_stories)

    def fetch(self, limit: int = 3) -> List[Story]:
        """Execute the main workflow."""
        best_stories = self.get_best_stories(limit=limit)
        return self.summarize_stories(best_stories)


def main() -> None:
    if len(sys.argv) != 1:
        print("Usage: python hacker_news.py")
        sys.exit(1)
    HackerNews().show(limit=3)


if __name__ == "__main__":
    main()
