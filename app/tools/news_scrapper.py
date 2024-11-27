import datetime
import json
import requests
from peewee import IntegrityError
from bs4 import BeautifulSoup
from newspaper import Article
from playwright.sync_api import sync_playwright
import ell

import nltk
from tenacity import retry, stop_after_attempt, wait_exponential
nltk.download('punkt')

from .news import News

# Store and load articles/links into a sqlite db

@ell.tool()
def load_google_news():
    """load todays news articles from https://news.google.com/home?hl=en-NZ&gl=NZ&ceid=NZ:en, stores a list of news articles"""
    with sync_playwright() as p:
        browser = p.webkit.launch()
        page = browser.new_page()
        page.goto('https://news.google.com/home?hl=en-NZ&gl=NZ&ceid=NZ:en', wait_until='networkidle')

        html_doc = BeautifulSoup(page.content(), 'html.parser')
        html_articles = html_doc.find_all('article')

        page.close()
        print("number of google articles: ", len(html_articles))
        for article in html_articles:
            link = [x for x in article.find_all('a') if x.text]
            if len(link) > 0:
                link = link[0]
            if link.has_attr('href') and link['href'].startswith('./read') and link.text != '' and 'quiz' not in link.text:
                link['href'] = f"https://news.google.com{link['href'].replace('./', '/')}"
                page = browser.new_page()
                page.goto(link['href'], wait_until='domcontentloaded')

                index = 0
                while page.url.startswith('https://news.google.com') and index < 3:
                    page.wait_for_load_state('networkidle')
                    index = index + 1

                # ignore pages that do not load
                if page.url.startswith('https://news.google.com'):
                    continue
                # Skip one word title
                if len(page.title()) < 2:
                    continue

                url = page.url
                print(page.content())
                page.close()

                try:
                    news = build_news_article(url)
                    insert_news(news, "google")
                except Exception as e:
                    continue

        browser.close()

@ell.tool()
def load_stuff_news():
    """load todays news articles from https://www.stuff.co.nz, stores a list of news articles"""
    # loader = PlaywrightURLLoader(urls=['https://www.stuff.co.nz'], remove_selectors=["header", "footer"])
    # html_doc = loader.load()
    return 'Still being implemented'

@ell.tool()
def load_one_news():
    """load todays news articles from https://www.1news.co.nz, stores a list of news articles"""
    with sync_playwright() as p:
        browser = p.webkit.launch()
        page = browser.new_page()
        page.goto('https://www.1news.co.nz', wait_until='domcontentloaded')

        html_doc = BeautifulSoup(page.content(), 'html.parser')
        html_articles = html_doc.find_all('div', class_='story')
        print("number of 1news articles: ", len(html_articles))
        for article in html_articles:
            link = [x for x in article.find_all('a') if x.find('h2') or x.find('h3')]
            if len(link) > 0:
                link = link[0]

            if link.has_attr('href'):
                title = (link.find('h2') or link.find('h3')).text
                # ignore Quiz, activities and classified
                if 'Quiz' in title:
                    continue

                try:
                    news_url = link['href']
                    if not news_url.startswith('https://www.1news.co.nz'):
                        news_url = f"https://www.1news.co.nz{news_url}"
                    news = build_news_article(news_url)
                    insert_news(news, "1news")
                except Exception as e:
                    continue

@ell.tool()
def load_nzherald_news():
    """load todays news articles from https://www.nzherald.co.nz, stores a list of news articles"""
    response = requests.get('https://www.nzherald.co.nz')

    html_doc = BeautifulSoup(response.text, 'html.parser')
    html_articles = html_doc.find_all('article', class_='story-card')
    print("number of nzherald articles: ", len(html_articles))
    for article in html_articles:
        link = [x for x in article.find_all('a') if x.find('h2') or x.find('h3')]
        if len(link) > 0:
            link = link[0]

        if link.has_attr('href'):
            # ignore oneroof articles
            if link['href'].startswith('https://www.oneroof.co.nz'):
                # print('ignore oneroof article: ', link['href'])
                continue

            # ignore drivencarguide articles
            if link['href'].startswith('https://www.drivencarguide.co.nz'):
                # print('ignore drivencarguide article: ', link['href'])
                continue

            # ignore businessdesk articles
            if link['href'].startswith('https://businessdesk.co.nz'):
                # print('ignore businessdesk article: ', link['href'])
                continue

            title = (link.find('h2') or link.find('h3')).text

            # ignore quiz, activities and classified
            if 'quiz' in title or 'Sudoku' in title or 'Crosswords' in title or 'classified ad' in title or title == 'Public Notices' or title == 'Death Notices':
                # print('ignore activities or classified: ', title)
                continue

            try:
                news = build_news_article(link['href'])
                insert_news(news, "nzherald")
            except Exception:
                continue
        else:
            breakpoint()
            print('')

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.8, min=60, max=600))
def build_news_article(url: str) -> Article:
    try:
        news = Article(url)
        news.build()
        return news
    except Exception as e:
        raise e


def insert_news(news: Article, source: str):
    try:
        date = news.publish_date
        if date is None or date == "":
            date = datetime.date.today()
        News.create(
            date=date,
            title=news.title,
            url=news.url,
            keywords=json.dumps(news.keywords),
            content=news.text,
            summary=news.summary,
            source=source,
        ).save()
        print(f"{source} news: ", news.title)
    except IntegrityError:
        pass
    except Exception as e:
        print(f"Error inserting {source} news: {e}")
