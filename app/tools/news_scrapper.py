import datetime
import json
import requests
from peewee import IntegrityError
from bs4 import BeautifulSoup
from newspaper import Article
from playwright.sync_api import sync_playwright
import ell

from .news import News

def build_news_article(url: str) -> Article:
    """Build a basic news article from a URL."""
    try:
        news = Article(url)
        news.download()
        news.parse()

        # Get text and handle empty content
        if not news.text:
            return news

        # Run newspaper's basic NLP
        news.nlp()
        return news
    except Exception as e:
        raise e

def insert_news(news: Article, source: str):
    """Insert a news article into the database."""
    try:
        date = news.publish_date or datetime.date.today()

        if news.text == '':
            return

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

@ell.tool()
def load_google_news():
    """load todays news articles from https://news.google.com/home?hl=en-NZ&gl=NZ&ceid=NZ:en"""
    with sync_playwright() as p:
        browser = p.webkit.launch()
        page = browser.new_page()
        page.goto('https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRFZxYUdjU0JXVnVMVWRDR2dKT1dpZ0FQAQ?ceid=NZ:en&oc=3', wait_until='domcontentloaded')

        prev_height = -1
        max_scrolls = 100
        scroll_count = 0

        while scroll_count < max_scrolls:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(3000)  # Adjust timeout as needed

            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                break

            prev_height = new_height
            scroll_count += 1

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

                try:
                    page = browser.new_page()
                    page.goto(link['href'], wait_until='domcontentloaded')

                    index = 0
                    while page.url.startswith('https://news.google.com') and index < 3:
                        page.wait_for_load_state('networkidle')
                        index = index + 1
                except Exception as e:
                    print(f"Error loading page: {e}")
                    continue

                # ignore pages that do not load
                if page.url.startswith('https://news.google.com'):
                    continue
                # Skip one word title
                if len(page.title()) < 2:
                    continue

                if page.title().startswith('Watch:'):
                    continue

                url = page.url
                page.close()

                try:
                    news = build_news_article(url)
                    insert_news(news, "google")
                except Exception as e:
                    continue

        browser.close()
        return "Successfully loaded google news"

@ell.tool()
def load_stuff_news():
    """load todays news articles from https://www.stuff.co.nz"""
    # loader = PlaywrightURLLoader(urls=['https://www.stuff.co.nz'], remove_selectors=["header", "footer"])
    # html_doc = loader.load()
    return 'Still being implemented'

@ell.tool()
def load_one_news():
    """load todays news articles from https://www.1news.co.nz"""
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
                if ('Quiz' in title or
                    'Full video' in title or
                    'UFC' in title or
                    'MMA' in title or
                    'WWE' in title or
                    'cartoon' in title or
                    title.strip().startswith('Watch:') or
                    title.strip().startswith('Full Video:') or
                    title.strip().startswith('Live updates:')):
                    continue

                try:
                    news_url = link['href']
                    if not news_url.startswith('https://www.1news.co.nz'):
                        news_url = f"https://www.1news.co.nz{news_url}"
                    news = build_news_article(news_url)
                    insert_news(news, "1news")
                except Exception as e:
                    continue

        browser.close()
        return "Successfully loaded 1news news"

@ell.tool()
def load_nzherald_news():
    """load todays news articles from https://www.nzherald.co.nz"""
    response = requests.get('https://www.nzherald.co.nz')

    html_doc = BeautifulSoup(response.text, 'html.parser')
    html_articles = html_doc.find_all('article', class_='story-card')
    print("number of nzherald articles: ", len(html_articles))
    for article in html_articles:
        link = [x for x in article.find_all('a') if x.find('h2') or x.find('h3')]
        if isinstance(link, list) and len(link) > 0:
            link = link[0]

        if isinstance(link, list):
            continue

        if link.has_attr('href'):
            # ignore oneroof articles
            if link['href'].startswith('https://www.oneroof.co.nz'):
                continue

            # ignore drivencarguide articles
            if link['href'].startswith('https://www.drivencarguide.co.nz'):
                continue

            # ignore businessdesk articles
            if link['href'].startswith('https://businessdesk.co.nz'):
                continue

            title = (link.find('h2') or link.find('h3')).text

            # ignore quiz, activities and classified
            if ('quiz' in title.lower() or
                'sudoku' in title.lower() or
                'crosswords' in title.lower() or
                'classified ad' in title or
                title == 'Public Notices' or
                title == 'Death Notices' or
                title == 'Book your ad online' or
                'Herald Premium' in title or
                'NZ Herald Live' in title or
                'UFC' in title or
                'MMA' in title or
                'WWE' in title or
                'cartoon' in title or
                title.strip().startswith('Video:') or
                title.strip().startswith('Livestream:') or
                title.strip().startswith('NZ Herald comments:') or
                title.strip().startswith('Watch:')):
                continue

            try:
                news = build_news_article(link['href'])
                insert_news(news, "nzherald")
            except Exception:
                continue

    return "Successfully loaded nzherald news"
