import os
import sys

import openai
import requests

from llm import LLM
from log import configure_logging

logger = configure_logging()


class HackerNews:
    TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
    ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"

    def __init__(self):
        self.model = "gpt-4o-mini_v2024-07-18"
        self.llm = LLM(self.model)

    def get_top_stories(self, limit=20):
        try:
            response = requests.get(self.TOP_STORIES_URL)
            response.raise_for_status()  # Raise an exception for HTTP errors
            top_story_ids = response.json()[:limit]

            top_stories = []
            for story_id in top_story_ids:
                story_response = requests.get(self.ITEM_URL.format(story_id))
                story_response.raise_for_status()  # Raise an exception for HTTP errors
                story_details = story_response.json()
                top_stories.append(story_details)
                logger.info(f"Fetched story: {story_details.get('title', 'No title')}")

            return top_stories

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching top stories: {e}")
            return []  # Return an empty list on error

    def summarize_text(self, text):
        response = openai.Completion.create(
            engine=self.model,
            prompt=f"Please summarize the following text:\n\n{text}\n\nSummary:",
            max_tokens=150,
        )
        summary = response.choices[0].text.strip()
        return summary

    def summarize_stories(self, stories):
        summaries = []
        for story in stories:
            title = story.get("title", "No title")
            url = story.get("url", "No URL")
            text = f"Title: {title}\nURL: {url}"

            # Fetch the story text if available
            if "text" in story:
                text += f"\n\nContent: {story['text']}"

            print(f"Summarizing story: {title}")
            summary = self.llm.summarize(text)
            summaries.append({"title": title, "url": url, "summary": summary})

        return summaries

    def display_summaries(self, summaries):
        for i, summary in enumerate(summaries, start=1):
            print(f"\nStory {i}")
            print(f"Title: {summary['title']}")
            print(f"URL: {summary['url']}")
            print(f"Summary: {summary['summary']}")


def main():
    if len(sys.argv) != 1:
        print("Usage: python hacker_news.py")
        sys.exit(1)

    hn = HackerNews()

    print("Fetching top 20 Hacker News stories...")
    top_stories = hn.get_top_stories(3)
    summaries = hn.summarize_stories(top_stories)
    hn.display_summaries(summaries)


if __name__ == "__main__":

    main()
