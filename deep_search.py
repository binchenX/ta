import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from log import configure_logging

logger = configure_logging()

# Initialize OpenAI client with validation
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("No OpenAI API key provided or found in env (OPENAI_API_KEY).")

# Model to use
MODEL = "gpt-4o-mini_v2024-07-18"

# Concurrency limit
CONCURRENCY_LIMIT = 2


# Trim text
def trim_prompt(text: str, max_chars: int) -> str:
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


# Web search
def search_web(query: str, limit: int = 5) -> List[Dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for g in soup.select("div.g")[:limit]:
        link = g.select_one("a")["href"] if g.select_one("a") else None
        if link and link.startswith("http"):
            try:
                page = requests.get(link, headers=headers, timeout=10)
                page_soup = BeautifulSoup(page.text, "html.parser")
                content = page_soup.get_text(separator="\n", strip=True)[:25000]
                results.append({"url": link, "markdown": content})
            except Exception as e:
                logger.error(f"Failed to scrape {link}: {e}")
    return results


# Generate SERP queries
def generate_serp_queries(
    query: str, num_queries: int = 3, learnings: List[str] = []
) -> List[Dict[str, str]]:
    prompt = f"""
    Given the user prompt: <prompt>{query}</prompt>
    Generate a list of up to {num_queries} unique SERP queries to research the topic.
    {f"Use these previous learnings to refine the queries: {chr(10).join(learnings)}" if learnings else ""}
    Return the result as a JSON object with a 'queries' key, where each query has a 'query' and 'researchGoal' field, e.g., 
    {{"queries": [{{"query": "example search", "researchGoal": "explore trends"}}]}}. Return only the JSON, nothing else.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise JSON generator. Return only valid JSON, no extra text.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,  # Increased to ensure complete JSON
    )
    raw_content = response.choices[0].message.content
    logger.info(f"Raw SERP queries response: {raw_content}")
    try:
        data = json.loads(raw_content)
        queries = data.get("queries", [])
        logger.info(f"Created {len(queries)} queries: {queries}")
        return queries[:num_queries]
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse SERP queries JSON: {e}")
        return []


# Process SERP results
def process_serp_result(
    query: str, result: List[Dict[str, str]], num_learnings: int = 3, num_follow_ups: int = 3
) -> Dict[str, List[str]]:
    contents = [trim_prompt(r["markdown"], 25000) for r in result]
    logger.info(f"Ran {query}, found {len(contents)} contents")
    prompt = f"""
    Given the SERP query: <query>{query}</query>
    And these contents: <contents>{chr(10).join([f"<content>{c}</content>" for c in contents])}</contents>
    Generate up to {num_learnings} unique, concise learnings and up to {num_follow_ups} follow-up questions.
    Learnings should be detailed, including entities, metrics, and dates where applicable.
    Return the result as a JSON object with 'learnings' and 'followUpQuestions' keys, e.g., 
    {{"learnings": ["fact 1"], "followUpQuestions": ["question 1"]}}. Return only the JSON, nothing else.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise JSON generator. Return only valid JSON, no extra text.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,  # Increased for more content
        timeout=60,
    )
    raw_content = response.choices[0].message.content
    logger.info(f"Raw SERP result response for {query}: {raw_content}")
    try:
        data = json.loads(raw_content)
        learnings = data.get("learnings", [])
        follow_ups = data.get("followUpQuestions", [])
        logger.info(f"Created {len(learnings)} learnings: {learnings}")
        return {
            "learnings": learnings[:num_learnings],
            "followUpQuestions": follow_ups[:num_follow_ups],
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse SERP result JSON for {query}: {e}")
        return {"learnings": [], "followUpQuestions": []}


# Generate final report
def write_final_report(prompt: str, learnings: List[str], visited_urls: List[str]) -> str:
    learnings_str = trim_prompt(
        chr(10).join([f"<learning>{l}</learning>" for l in learnings]), 150000
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise JSON generator. Return only valid JSON, no extra text.",
            },
            {
                "role": "user",
                "content": f"""
        Given the user prompt: <prompt>{prompt}</prompt>
        And these research learnings: <learnings>{learnings_str}</learnings>
        Write a detailed Markdown report (aim for 3+ pages) including all learnings.
        Return the result as a JSON object with a 'reportMarkdown' key, e.g., {{"reportMarkdown": "# Report\nContent..."}}. Return only the JSON, nothing else.
        """,
            },
        ],
        max_tokens=1500,  # Increased for longer report
    )
    raw_content = response.choices[0].message.content
    logger.info(f"Raw report response: {raw_content}")
    try:
        report = json.loads(raw_content)["reportMarkdown"]
        sources = "\n\n## Sources\n\n" + chr(10).join([f"- {url}" for url in visited_urls])
        return report + sources
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse report JSON: {e}")
        return "# Error\nFailed to generate report due to JSON parsing error."


# Deep research function
def deep_research(
    query: str,
    breadth: int = 4,
    depth: int = 2,
    learnings: List[str] = [],
    visited_urls: List[str] = [],
    on_progress=None,
) -> Dict[str, List[str]]:
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

    serp_queries = generate_serp_queries(query, breadth, learnings)
    report_progress(
        {
            "totalQueries": len(serp_queries),
            "currentQuery": serp_queries[0]["query"] if serp_queries else "",
        }
    )

    def process_query(serp_query: Dict[str, str]) -> Dict[str, List[str]]:
        try:
            result = search_web(serp_query["query"])
            new_urls = [r["url"] for r in result]
            processed = process_serp_result(
                serp_query["query"], result, num_follow_ups=max(1, breadth // 2)
            )
            all_learnings = learnings + processed["learnings"]
            all_urls = visited_urls + new_urls
            new_depth = depth - 1
            new_breadth = max(1, breadth // 2)

            if new_depth > 0:
                logger.info(f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}")
                next_query = f"Previous goal: {serp_query['researchGoal']}\nFollow-ups:{chr(10).join(processed['followUpQuestions'])}"
                report_progress(
                    {
                        "currentDepth": new_depth,
                        "currentBreadth": new_breadth,
                        "completedQueries": progress["completedQueries"] + 1,
                        "currentQuery": serp_query["query"],
                    }
                )
                return deep_research(
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
            logger.error(f"Error processing {serp_query['query']}: {e}")
            return {"learnings": [], "visitedUrls": []}

    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        results = list(executor.map(process_query, serp_queries))

    all_learnings = list(set(sum((r["learnings"] for r in results), [])))
    all_urls = list(set(sum((r["visitedUrls"] for r in results), [])))
    return {"learnings": all_learnings, "visitedUrls": all_urls}


# Main function with hardcoded query and defaults
def main():
    query = "coolest tech in 2025"
    breadth = 4  # Default
    depth = 2  # Default

    def progress_callback(p):
        logger.info(
            f"Depth: {p['currentDepth']}/{p['totalDepth']}, Breadth: {p['currentBreadth']}/{p['totalBreadth']}, Query {p['completedQueries']}/{p['totalQueries']} - {p.get('currentQuery', '')}"
        )

    logger.info(f"Starting research on: {query}")
    result = deep_research(query, breadth, depth, on_progress=progress_callback)
    report = write_final_report(query, result["learnings"], result["visitedUrls"])

    with open("output.md", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to output.md")


if __name__ == "__main__":
    main()
