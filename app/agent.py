import asyncio
from datetime import date, timedelta, datetime
import os
import random
from time import sleep
import traceback
from openai import Client, RateLimitError
from os import getenv
import ell
import json
from itertools import islice
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

from tools.news_processor import NewsProcessor
from tools.news_scrapper import load_one_news, load_nzherald_news, load_google_news
from tools.news import News, NewsSummary, NewsGroup
from tools.retry import with_retry
from tools.news_grouper import KeywordBasedNewsGrouper

ell.init(store='./logdir', autocommit=True, verbose=False)

groq_model = "llama-3.1-8b-instant"
groq_client = Groq(api_key=getenv("GROQ_API_KEY"))


News.create_table()
NewsSummary.create_table()
NewsGroup.create_table()

tools = [load_one_news, load_nzherald_news, load_google_news]

@with_retry
@ell.complex(model=groq_model, tools=tools, temperature=0.1, client=groq_client)
def initiate_daily():
    """You are a helpful assistant that has tools to get news from different sources."""
    return "get all news from one news and nzherald"

@with_retry
@ell.simple(model=groq_model, temperature=0.1, client=groq_client)
def pick_interesting_news_groups(groups: list[dict]) -> str:
    """You are an experienced journalist that picks out the most interesting news groups."""
    return f"""
    From the following list of news groups from the last 3 days, pick out THE MOST interesting and important news groups.
    Each group contains related articles about the same topic or event.

    STRICT REQUIREMENT: Select MAXIMUM 2 groups total.

    Selection criteria:
    1. Most impactful or significant to society
    2. Breaking news or major developments
    3. Stories that people should know about
    4. Diverse topics (don't pick multiple groups about the same general topic)
    5. Prioritize more recent news (within the last 3 days)

    Chain of thought process:
    1. Identify the most impactful stories based on selection criteria
    2. From these, identify which are truly breaking news or major developments
    3. Consider which stories the public needs to know about most urgently
    4. Finally, select THE MOST important 1-2 stories that meet these criteria

    The groups are formatted as [{{
        "title": "title of the group",
        "article_count": number of articles,
        "keywords": ["keyword1", "keyword2"],
        "date": "YYYY-MM-DD"
    }}, ...].

    <news_groups>
    {json.dumps(groups, indent=2)[:18000]}
    </news_groups>

    Return ONLY the titles of the 1-2 most important groups in a JSON array format: ["title1", "title2"]
    We only want the json array and nothing else.
    You MUST return 1 or 2 groups maximum - no exceptions.
    """

def process_groups_in_batches(group_info: list[dict], batch_size: int = 5) -> list[str]:
    """Process groups in batches to avoid token limits."""
    interesting_groups = set()
    total_batches = (len(group_info) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        batch = list(islice(group_info, start_idx, start_idx + batch_size))

        try:
            print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch)} groups)")
            selected_groups = json.loads(pick_interesting_news_groups(batch))
            interesting_groups.update(selected_groups)
            print(f"Selected {len(selected_groups)} groups from this batch")

            # Add a small delay between batches
            if batch_num < total_batches - 1:
                asyncio.run(asyncio.sleep(2))

        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {str(e)}")
            print(traceback.format_exc())
            # If there's an error, include all groups from this batch
            interesting_groups.update(g["title"] for g in batch)

    return list(interesting_groups)

@with_retry
@ell.simple(client=groq_client, model=groq_model, temperature=0.9)
def summarise_news(news: dict):
    """You are a experienced journalist that summarises news articles into points."""
    return f"""
    From the following given news, summarize it into points, make sure to format it as markdown.
    The lists of news are formatted as {{ "title": "content" }}.

    Just return the summary in the following format:
    * Point 1
    * Point 2
    * Point 3
    ...

    --- news ---
    {json.dumps(news, indent=2)[:20000]}
    """

def get_list_of_summarised_news(grouped_news: dict):
    for keywords, news_ids in grouped_news.items():
        news = News.select().where(News.id.in_(news_ids))
        main_title = min(news, key=lambda x: len(x.title)).title
        content = "\n\n".join([n.content for n in news])

        try:
            result = summarise_news({main_title: content})
            print(f"Summarised {main_title}")
        except RateLimitError as e:
            print(f"Rate limit error: {str(e)}")
            raise  # Re-raise to trigger retry
        except Exception as e:
            print(f"Error summarizing news: {str(e)}")
            print(traceback.format_exc())
            continue

        result = result.replace('"', "")
        try:
            NewsSummary.replace(title=main_title, summary=result, keywords=keywords).execute()
        except Exception as e:
            print(f"Error saving summary: {str(e)}")
            print(traceback.format_exc())

def wait_random_time():
    current_time = datetime.now()
    print(f"Current time: {current_time}")
    wait_till = current_time + timedelta(hours=1.8) + timedelta(seconds=random.uniform(0.1, 60)) # Add a random amount of time between 0.1 and 1 minute
    counter = 0
    while datetime.now() < wait_till:
        if counter % 50 == 0 and counter != 0: # every 5 seconds
            print(".", end="")
        if counter % 6000 == 0: # every 10 minutes
            print()
            print(f"Another {wait_till - datetime.now()} before initiating daily news")
        sleep(0.1)
        counter += 1
    print()

def main():
    while True:
        today = date.today()
        result = initiate_daily()
        if result.tool_calls:
            # This is done so that we can pass the tool calls to the language model
            result_message = result.call_tools_and_collect_as_message(parallel=True, max_workers=3)
            print("Message to be sent to the LLM:\n", result_message.text)

            NewsProcessor.process_all_unprocessed()

            # Use the new NewsGrouper to process articles
            grouper = KeywordBasedNewsGrouper()
            total_processed = grouper.group_all_ungrouped()
            print(f"Processed {total_processed} articles into groups")

            # Get all groups created
            groups = NewsGroup.select().where(NewsGroup.date >= today - timedelta(days=3))
            print(f"Found {len(groups)} groups from the last 3 days")

            defined_groups = [group for group in groups if group.articles.count() > 1]
            print(f"Found {len(defined_groups)} defined groups from the last 3 days")

            # Prepare groups for interesting news selection
            group_info = []
            grouped_news = {}

            for group in defined_groups:
                news_in_group = News.select().where(News.group == group)
                news_count = len(list(news_in_group))
                # Parse keywords from string to list
                try:
                    keywords = json.loads(group.keywords)
                except json.JSONDecodeError:
                    keywords = group.keywords.split(", ")

                group_info.append({
                    "title": group.title,
                    "article_count": news_count,
                    "keywords": keywords,
                    "date": group.date.isoformat()  # Add date for reference
                })
                grouped_news[group.title] = {
                    "ids": [n.id for n in news_in_group],
                    "keywords": group.keywords,
                    "date": group.date.isoformat()
                }

            print(f"Found {len(group_info)} groups from the last 3 days")

            # Process groups in batches and get interesting ones
            interesting_groups = process_groups_in_batches(group_info)
            print(f"Selected {len(interesting_groups)} interesting groups in total")

            # Filter for interesting groups only
            interesting_grouped_news = {}
            for title in interesting_groups:
                if title in grouped_news:
                    interesting_grouped_news[grouped_news[title]["keywords"]] = grouped_news[title]["ids"]

            print(f"Grouped news to summarize: {json.dumps(interesting_grouped_news, indent=2)}")

            # Summarize the grouped news
            NewsSummary.delete().execute()
            get_list_of_summarised_news(interesting_grouped_news)

            # Write summaries to file
            summarised_news = NewsSummary.select().where(NewsSummary.date == today)
            filename = f"./output/{today.strftime('%Y-%m-%d')}.md"
            if os.path.exists(filename):
                os.remove(filename)

            if not os.path.exists("./output"):
                os.makedirs("./output")

            print(f"Writing to {filename}")
            with open(filename, "w") as f:
                f.write("# Today's News - " + today.strftime('%A, %d %B %Y') + " - Last updated: " + datetime.now().strftime('%H:%M %Z') + "\n\n")
                for sn in summarised_news:
                    f.write(f"### {sn.title}\n")
                    f.write(sn.summary)
                    f.write("\n-----\n\n")
        wait_random_time()

if __name__ == "__main__":
    main()
