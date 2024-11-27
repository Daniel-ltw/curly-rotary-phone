import asyncio
import datetime
import os
import re
import traceback
from openai import OpenAI, RateLimitError
from os import getenv
from tenacity import retry, stop_after_attempt
from peewee import IntegrityError
import ell
import json

from dotenv import load_dotenv
load_dotenv()


# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=getenv("GROQ_API_KEY"),
)

ell.init(store='./logdir', autocommit=True, verbose=False, default_client=client, autocommit_model="llama-3.1-8b-instant")

from tools.news_scrapper import load_google_news, load_stuff_news, load_nzherald_news, load_one_news

from tools.news import News, NewsSummary

News.create_table()
NewsSummary.create_table()


def parse_retry_time(exception):
    if "rate_limit_exceeded" in str(exception):
        match = re.search(r"Please try again in (\d+)m(\d+\.\d+)s", str(exception))
        if match:
            minutes = float(match.group(1))
            seconds = float(match.group(2)) + 1
            return minutes * 60 + seconds
        else:
            print(str(exception))
    return 60

def wait_strategy(retry_state):
    exception = retry_state.outcome.exception()
    wait_time = parse_retry_time(exception)
    return wait_time


# tools = [load_stuff_news, load_google_news, load_nzherald_news, load_one_news]
# tools = [load_google_news, load_nzherald_news]
tools = [load_one_news, load_nzherald_news]

@retry(stop=stop_after_attempt(8), wait=wait_strategy)
@ell.complex(model="llama-3.1-8b-instant", client=client, tools=tools, temperature=0.1)
def initiate_daily():
    """You are a helpful assistant that has tools to get news from different sources."""
    # return "get all news from nzherald and google"
    return "get all news from one news and nzherald"

@retry(stop=stop_after_attempt(8), wait=wait_strategy)
@ell.simple(model="llama-3.1-8b-instant", client=client, temperature=0.9)
def summarise_news(news: dict):
    """You are a experienced journalist that summarises news articles into points."""
    return f"""
    From the following given list of news, summarize them into points, make sure to format it as markdown.
    The lists of news are formatted as {{ "title": "content" }}.

    Just return the summary in the following format:
    * Point 1
    * Point 2
    * Point 3
    ...

    <news>
    {json.dumps(news, indent=2)[:20000]}
    </news>
    """

@retry(stop=stop_after_attempt(8), wait=wait_strategy)
@ell.simple(model="llama-3.1-8b-instant", client=client, temperature=0.1)
def pick_interesting_news(news: list[str]):
    """You are a experienced journalist that picks out the most interesting news."""
    return f"""
    From the following given list of news, pick out the most interesting news and return the title of the news.
    The lists of news are formatted as [{{ "title": "summary" }}, {{ "title": "summary" }}].

    <news>
    {json.dumps(news, indent=2)[:20000]}
    </news>

    Just return the title of the news in the following format: "title"
    """

async def get_list_of_interesting_news() -> list[str]:
    final_news = []
    prepared_news = []
    news = News.select().order_by(News.id.desc())
    # Select the top 10 news?
    for index, n in enumerate(news):
        if index != 0 and index % 8 == 0:
            await asyncio.sleep(20)
            try:
                result = pick_interesting_news(prepared_news)
            except RateLimitError as e:
                wait_time = parse_retry_time(e)
                print(f"Rate limit exceeded, waiting for {wait_time} seconds")
                await asyncio.sleep(wait_time)
                result = pick_interesting_news(prepared_news)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                continue
            try:
                final_news.append(json.loads(result))
            except json.decoder.JSONDecodeError:
                result = result.replace('"', "")
                final_news.append(result)
            prepared_news = []
        prepared_news.append(n.llm_summary_format())
    return final_news

async def get_list_of_summarised_news(final_news: list[str]):
    news = News.select().where(News.title.in_(final_news))
    for n in news:
        try:
            result = summarise_news(n.llm_content_format())
        except RateLimitError as e:
            wait_time = parse_retry_time(e)
            print(f"Rate limit exceeded, waiting for {wait_time} seconds")
            await asyncio.sleep(wait_time)
            result = summarise_news(n.llm_content_format())
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            continue

        result = result.replace('"', "")
        try:
            summary = NewsSummary.create(title=n.title, summary=result)
            summary.save()
        except IntegrityError:
            pass


def main():
    result = initiate_daily()
    print(result.text)
    if result.tool_calls:
        # This is done so that we can pass the tool calls to the language model
        result_message = result.call_tools_and_collect_as_message(parallel=True, max_workers=3)
        print("Message to be sent to the LLM:", result_message.text) # Representation of the message to be sent to the LLM.

        final_news = asyncio.run(get_list_of_interesting_news())
        asyncio.run(get_list_of_summarised_news(final_news))


        summarised_news = NewsSummary.select()
        filename = f"./output/{datetime.date.today().strftime('%Y-%m-%d')}.md"
        if os.path.exists(filename):
            os.remove(filename)

        if not os.path.exists("./output"):
            os.makedirs("./output")

        with open(filename, "w") as f:
            f.write("# Today's News\n\n")
            for sn in summarised_news:
                f.write(f"### {sn.title}\n")
                f.write(sn.summary)
                f.write("\n-----\n\n")

if __name__ == "__main__":
    main()


# TODO:
# - Need to be able to group news by 2-3 matching keywords
