import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import requests
from openai import OpenAI

from log import configure_logging


class DeepSearch:
    def __init__(self, openai_api_key: str = None, brave_api_key: str = None):
        self.logger = configure_logging()

        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("No OpenAI API key provided or found in env (OPENAI_API_KEY).")
        self.client = OpenAI(api_key=self.openai_api_key)

        # Initialize Brave Search
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        if not self.brave_api_key:
            raise ValueError("No Brave Search API key provided or found in env (BRAVE_API_KEY).")

        # Configuration
        self.MODEL = "gpt-4o-mini_v2024-07-18"
        self.CONCURRENCY_LIMIT = 2

    def trim_prompt(self, text: str, max_chars: int) -> str:
        """Trim text to specified maximum length."""
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    def search_web(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Perform web search using Brave Search API."""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": self.brave_api_key}
        params = {"q": query, "count": limit}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            results = data.get("web", {}).get("results", [])
            self.logger.info(f"Brave Search for '{query}' returned {len(results)} results")
            return [
                {"url": r["url"], "markdown": r.get("description", "")} for r in results[:limit]
            ]
        except requests.RequestException as e:
            self.logger.error(f"Brave Search failed for '{query}': {e}")
            return []

    def generate_serp_queries(
        self, query: str, num_queries: int = 3, learnings: List[str] = []
    ) -> List[Dict[str, str]]:
        """Generate search queries based on the input query and previous learnings."""
        prompt = f"""
        Given the user prompt: <prompt>{query}</prompt>
        Generate a list of up to {num_queries} unique SERP queries to research the topic.
        {f"Use these previous learnings to refine the queries: {chr(10).join(learnings)}" if learnings else ""}
        Return the result as a JSON object with a 'queries' key, where each query has a 'query' and 'researchGoal' field.
        """

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise JSON generator. Return only valid JSON, no extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )

        try:
            data = json.loads(response.choices[0].message.content)
            queries = data.get("queries", [])
            self.logger.info(f"Created {len(queries)} queries: {queries}")
            return queries[:num_queries]
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse SERP queries JSON: {e}")
            return []

    def process_serp_result(
        self,
        query: str,
        result: List[Dict[str, str]],
        num_learnings: int = 3,
        num_follow_ups: int = 3,
    ) -> Dict[str, List[str]]:
        """Process search results to extract learnings and follow-up questions."""
        contents = [self.trim_prompt(r["markdown"], 25000) for r in result]
        self.logger.info(f"Processing {len(contents)} results for query: {query}")

        prompt = f"""
        Given the SERP query: <query>{query}</query>
        And these contents: <contents>{chr(10).join([f"<content>{c}</content>" for c in contents])}</contents>
        Generate up to {num_learnings} unique, concise learnings and up to {num_follow_ups} follow-up questions.
        Return as JSON with 'learnings' and 'followUpQuestions' arrays.
        """

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise JSON generator. Return only valid JSON, no extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            timeout=60,
        )

        try:
            data = json.loads(response.choices[0].message.content)
            return {
                "learnings": data.get("learnings", [])[:num_learnings],
                "followUpQuestions": data.get("followUpQuestions", [])[:num_follow_ups],
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse SERP result JSON for {query}: {e}")
            return {"learnings": [], "followUpQuestions": []}

    def write_final_report(self, prompt: str, learnings: List[str], visited_urls: List[str]) -> str:
        """Generate final markdown report from research findings."""
        learnings_str = self.trim_prompt(
            chr(10).join([f"<learning>{l}</learning>" for l in learnings]), 150000
        )

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise JSON generator. Return only valid JSON, no extra text.",
                },
                {
                    "role": "user",
                    "content": f"""
                Generate a detailed Markdown report for the query: {prompt}
                Based on these learnings: {learnings_str}
                Include an executive summary, key findings, and detailed analysis.
                Return as JSON with a 'reportMarkdown' key.
                """,
                },
            ],
            max_tokens=1500,
        )

        try:
            report = json.loads(response.choices[0].message.content)["reportMarkdown"]
            sources = "\n\n## Sources\n\n" + chr(10).join([f"- {url}" for url in visited_urls])
            return report + sources
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse report JSON: {e}")
            return "# Error\nFailed to generate report due to JSON parsing error."

    def deep_research(
        self,
        query: str,
        breadth: int = 2,
        depth: int = 2,
        learnings: List[str] = [],
        visited_urls: List[str] = [],
        on_progress: Optional[Callable] = None,
    ) -> Dict[str, List[str]]:
        """Perform deep recursive research on a topic."""
        progress = {
            "currentDepth": depth,
            "totalDepth": depth,
            "currentBreadth": breadth,
            "totalBreadth": breadth,
            "totalQueries": 0,
            "completedQueries": 0,
        }

        def report_progress(update: Dict):
            progress.update(update)
            if on_progress:
                on_progress(progress)

        serp_queries = self.generate_serp_queries(query, breadth, learnings)
        report_progress(
            {
                "totalQueries": len(serp_queries),
                "currentQuery": serp_queries[0]["query"] if serp_queries else "",
            }
        )

        def process_query(serp_query: Dict[str, str]) -> Dict[str, List[str]]:
            try:
                result = self.search_web(serp_query["query"])
                new_urls = [r["url"] for r in result]
                processed = self.process_serp_result(
                    serp_query["query"], result, num_follow_ups=max(1, breadth // 2)
                )

                all_learnings = learnings + processed["learnings"]
                all_urls = visited_urls + new_urls
                new_depth = depth - 1
                new_breadth = max(1, breadth // 2)

                if new_depth > 0:
                    self.logger.info(
                        f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}"
                    )
                    next_query = f"Previous goal: {serp_query['researchGoal']}\nFollow-ups:{chr(10).join(processed['followUpQuestions'])}"

                    report_progress(
                        {
                            "currentDepth": new_depth,
                            "currentBreadth": new_breadth,
                            "completedQueries": progress["completedQueries"] + 1,
                            "currentQuery": serp_query["query"],
                        }
                    )

                    return self.deep_research(
                        next_query, new_breadth, new_depth, all_learnings, all_urls, on_progress
                    )
                else:
                    report_progress(
                        {
                            "currentDepth": 0,
                            "completedQueries": progress["completedQueries"] + 1,
                            "currentQuery": serp_query["query"],
                        }
                    )
                    return {"learnings": all_learnings, "visitedUrls": all_urls}

            except Exception as e:
                self.logger.error(f"Error processing {serp_query['query']}: {e}")
                return {"learnings": [], "visitedUrls": []}

        with ThreadPoolExecutor(max_workers=self.CONCURRENCY_LIMIT) as executor:
            results = list(executor.map(process_query, serp_queries))

        all_learnings = list(set(sum((r["learnings"] for r in results), [])))
        all_urls = list(set(sum((r["visitedUrls"] for r in results), [])))

        return {"learnings": all_learnings, "visitedUrls": all_urls}

    def research(
        self, query: str, breadth: int = 2, depth: int = 2, on_progress: Optional[Callable] = None
    ) -> str:
        """
        Main public method to perform research and generate report.

        Args:
            query (str): The research query
            breadth (int): Number of parallel search paths
            depth (int): How deep to follow up on each path
            on_progress (Callable, optional): Callback for progress updates

        Returns:
            str: Markdown formatted research report
        """
        self.logger.info(f"Starting research on: {query}")
        result = self.deep_research(query, breadth, depth, on_progress=on_progress)
        report = self.write_final_report(query, result["learnings"], result["visitedUrls"])
        return report


def main():
    searcher = DeepSearch()

    def progress_callback(p):
        print(
            f"Progress: Depth {p['currentDepth']}/{p['totalDepth']}, "
            f"Breadth {p['currentBreadth']}/{p['totalBreadth']}, "
            f"Queries {p['completedQueries']}/{p['totalQueries']}"
        )

    report = searcher.research(
        query="coolest tech in 2025", breadth=2, depth=2, on_progress=progress_callback
    )

    with open("output.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Report saved to output.md")


if __name__ == "__main__":
    main()
