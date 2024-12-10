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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np

from .news import News

# Store and load articles/links into a sqlite db

TOPIC_KEYWORDS = {
    'road_trip': {
        'primary': ['road trip'],
        'related': ['journey', 'travel', 'destination']
    },
    'fire_emergency': {
        'primary': ['fire', 'fenz'],
        'related': ['emergency', 'crews', 'evacuate']
    },
    'syria_conflict': {
        'primary': ['syria', 'assad'],
        'related': ['rebel', 'regime', 'damascus']
    },
    'formula1': {
        'primary': ['f1', 'grand prix'],
        'related': ['racing', 'driver']
    },
    'youth_programs': {
        'primary': ['boot camp'],
        'related': ['youth', 'teenager']
    }
}

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
                if 'Quiz' in title or 'Full video' in title or title.startswith(r' ?Watch:'):
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
            if 'quiz' in title or 'Sudoku' in title or 'Crosswords' in title or 'classified ad' in title or title == 'Public Notices' or title == 'Death Notices' or 'Herald Premium' in title or 'NZ Herald Live' in title or title.startswith(r' ?Watch:'):
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

def build_news_article(url: str) -> Article:
    try:
        news = Article(url)
        news.download()
        news.parse()

        # Get text and handle empty content
        text = news.text
        if not text:
            return news

        # Create TF-IDF vectorizer for keyword extraction
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10,
            ngram_range=(1, 2)
        )

        # Extract keywords using both TF-IDF and newspaper's implementation
        news.nlp()  # This runs newspaper's implementation
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Get TF-IDF keywords
        tfidf_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-5:][::-1]]

        # Check for topic keywords in title and content
        title_and_text = f"{news.title.lower()} {text.lower()}"
        topic_keywords = []
        matched_topics = []

        for topic, keywords in TOPIC_KEYWORDS.items():
            # Check for primary keywords (strong indicators)
            primary_matches = sum(1 for k in keywords['primary']
                                if k in title_and_text)
            # Check for related keywords
            related_matches = sum(1 for k in keywords['related']
                                if k in title_and_text)

            # Add topic if we have strong matches
            if primary_matches > 0 or related_matches >= 2:
                topic_keywords.extend(keywords['primary'] + keywords['related'])
                matched_topics.append(topic)

        # Add matched topics to keywords with weights
        all_keywords = list(set(
            news.keywords[:5] +  # Original newspaper keywords
            tfidf_keywords +     # TF-IDF keywords
            topic_keywords       # Topic-specific keywords
        ))

        news.keywords = all_keywords

        # Enhanced summarization
        sentences = sent_tokenize(text)

        if len(sentences) > 3:
            # Score sentences based on keyword presence and position
            sent_scores = []
            for i, sentence in enumerate(sentences):
                # Position score - earlier sentences get higher weight
                position_score = 1.0 / (i + 1)

                # Keyword score - now using our enhanced keywords
                keyword_score = sum(1 for keyword in news.keywords
                                  if keyword.lower() in sentence.lower())

                # Topic relevance score
                topic_score = sum(1 for keyword in topic_keywords
                                if keyword in sentence.lower())

                # Combined score with topic relevance
                total_score = (keyword_score * 0.5) + (position_score * 0.3) + (topic_score * 0.2)
                sent_scores.append((sentence, total_score))

            # Get top 3 sentences
            sent_scores.sort(key=lambda x: x[1], reverse=True)
            enhanced_summary = ' '.join(sent[0] for sent in sent_scores[:3])

            # Always use enhanced summary
            news.summary = enhanced_summary

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
