from datetime import date
import json
import traceback
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .news import News, NewsGroup, db

class KeywordBasedNewsGrouper:
    def __init__(self,
                 min_keyword_match_ratio: float = 0.45,
                 keyword_similarity_threshold: float = 0.45,
                 title_similarity_threshold: float = 0.25,
                 summary_similarity_threshold: float = 0.45):
        """Initialize the KeywordBasedNewsGrouper."""
        self.min_keyword_match_ratio = min_keyword_match_ratio
        self.keyword_similarity_threshold = keyword_similarity_threshold
        self.title_similarity_threshold = title_similarity_threshold
        self.summary_similarity_threshold = summary_similarity_threshold

        # Initialize vectorizers
        self.title_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 4),
            token_pattern=r'(?u)\b\w+\b',
            max_features=5000
        )

        self.keyword_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )

        self.summary_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 5),  # Increased from (1, 3) to capture longer phrases
            max_features=5000,
            token_pattern=r'(?u)\b\w+\b'
        )

        # Initialize NLTK for word normalization
        from nltk.stem import WordNetLemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def _normalize_word(self, word: str) -> str:
        """Normalize a word by converting to singular form and lowercase."""
        word = word.lower().strip()
        word = self.lemmatizer.lemmatize(word, pos='n')
        word = self.lemmatizer.lemmatize(word, pos='v')
        return word

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        """Normalize a list of keywords."""
        normalized = []
        for keyword in keywords:
            words = keyword.split()
            if len(words) > 1:
                normalized.append(' '.join(self._normalize_word(w) for w in words))
            else:
                normalized.append(self._normalize_word(keyword))
        return normalized

    def _calculate_keyword_similarity(self, keywords1: List[str], keywords2: List[str]) -> tuple:
        """Calculate both exact matches and similarity between keyword sets."""
        # Normalize keywords
        norm_keywords1 = self._normalize_keywords(keywords1)
        norm_keywords2 = self._normalize_keywords(keywords2)

        # Count exact matches after normalization
        set1 = set(norm_keywords1)
        set2 = set(norm_keywords2)
        exact_matches = len(set1.intersection(set2))

        # Calculate cosine similarity
        combined_keywords = list(set1.union(set2))
        vectorizer = self.keyword_vectorizer.fit(combined_keywords)
        text1 = ' '.join(norm_keywords1)
        text2 = ' '.join(norm_keywords2)
        vectors = vectorizer.transform([text1, text2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        return exact_matches, similarity

    def _calculate_title_similarity(self, titles: List[str]) -> np.ndarray:
        """Calculate similarity matrix between article titles."""
        tfidf_matrix = self.title_vectorizer.fit_transform(titles)
        return cosine_similarity(tfidf_matrix)

    def _parse_keywords(self, keywords_str: str) -> List[str]:
        """Safely parse keywords string to list."""
        try:
            if isinstance(keywords_str, list):
                return keywords_str
            return json.loads(keywords_str)
        except json.JSONDecodeError:
            try:
                return [k.strip() for k in keywords_str.split(',')]
            except Exception as e:
                print(f"Error parsing keywords '{keywords_str}': {str(e)}")
                return []

    def _get_shortest_title(self, articles: List[News]) -> str:
        """Get the shortest title from a list of articles."""
        return min((article.title for article in articles), key=len)

    def _get_earliest_date(self, articles: List[News]) -> date:
        """Get the earliest date from a list of articles."""
        return min(article.date for article in articles)

    def _calculate_summary_similarity(self, summaries: List[str]) -> np.ndarray:
        """Calculate similarity matrix between article summaries."""
        # Create TF-IDF matrix for summaries
        try:
            tfidf_matrix = self.summary_vectorizer.fit_transform(summaries)
            return cosine_similarity(tfidf_matrix)
        except Exception as e:
            print(f"Error calculating summary similarity: {str(e)}")
            return np.zeros((len(summaries), len(summaries)))

    def _are_articles_related(self,
                            article1: News,
                            article2: News,
                            title_similarity_matrix: np.ndarray,
                            summary_similarity_matrix: np.ndarray,
                            article_indices: tuple) -> tuple[bool, dict]:
        """Determine if two articles are related based on keywords, title, and summary similarity."""
        # Get keywords
        keywords1 = self._parse_keywords(article1.keywords)
        keywords2 = self._parse_keywords(article2.keywords)

        # Calculate both exact matches and similarity for keywords
        exact_matches, keyword_similarity = self._calculate_keyword_similarity(keywords1, keywords2)

        # Get title and summary similarity
        i, j = article_indices
        title_similarity = title_similarity_matrix[i][j]
        summary_similarity = summary_similarity_matrix[i][j]

        # Ignore perfect or near-perfect matches as they might be duplicates
        if title_similarity > 0.98:
            title_similarity = 0.0
        if summary_similarity > 0.98:
            summary_similarity = 0.0

        # Calculate required minimum matches
        min_required_matches = max(1, round(len(keywords1) * self.min_keyword_match_ratio))

        # Check for common words in titles
        title_words1 = set(word.lower() for word in article1.title.split())
        title_words2 = set(word.lower() for word in article2.title.split())
        common_significant_words = title_words1.intersection(title_words2) - set(self.title_vectorizer.get_stop_words())

        # Boost title similarity for significant common words
        if len(common_significant_words) >= 2:
            title_similarity = max(title_similarity, 0.3)

        # Check if articles share keywords
        keywords1 = set(k.lower() for k in self._parse_keywords(article1.keywords))
        keywords2 = set(k.lower() for k in self._parse_keywords(article2.keywords))
        shared_keywords = keywords1.intersection(keywords2)

        # Boost keyword similarity based on shared keywords quality
        if shared_keywords:
            keyword_similarity *= (1 + (len(shared_keywords) / max(len(keywords1), len(keywords2))))

        # Articles are related if they meet these criteria:
        # 1. Have enough exact keyword matches OR high keyword similarity
        keyword_match = (exact_matches >= min_required_matches or
                        keyword_similarity >= self.keyword_similarity_threshold)

        # 2. Must have both:
        #    - Good title similarity AND
        #    - Some summary similarity
        content_match = (
            title_similarity >= self.title_similarity_threshold and
            summary_similarity >= self.summary_similarity_threshold
        )

        match_details = {
            'exact_keyword_matches': exact_matches,
            'keyword_similarity': round(keyword_similarity, 3),
            'title_similarity': round(title_similarity, 3),
            'summary_similarity': round(summary_similarity, 3),
            'min_required_matches': min_required_matches,
            'matched_by': []
        }

        if exact_matches >= min_required_matches:
            match_details['matched_by'].append('exact_keywords')
        if keyword_similarity >= self.keyword_similarity_threshold:
            match_details['matched_by'].append('keyword_similarity')
        if title_similarity >= self.title_similarity_threshold:
            match_details['matched_by'].append('title_similarity')
        if summary_similarity >= self.summary_similarity_threshold:
            match_details['matched_by'].append('summary_similarity')

        return keyword_match and content_match, match_details

    def process_ungrouped_articles(self, batch_size: int = 100):
        """Process ungrouped articles using keyword matching, title and summary similarity.

        Args:
            batch_size: Maximum number of articles to process in one batch
        """
        try:
            # Get ungrouped articles
            ungrouped_articles = list(News
                                    .select()
                                    .where(News.group.is_null())
                                    .order_by(News.id.desc())
                                    .limit(batch_size))

            if not ungrouped_articles:
                print("No ungrouped articles found")
                return 0

            print(f"\nProcessing batch of {len(ungrouped_articles)} articles")

            # Get all existing groups for comparison
            existing_groups = list(NewsGroup.select().order_by(NewsGroup.id.desc()))

            # Calculate title similarity matrix for ungrouped articles
            titles = [article.title for article in ungrouped_articles]
            title_similarity_matrix = self._calculate_title_similarity(titles)

            # Calculate summary similarity matrix
            summaries = [article.summary for article in ungrouped_articles]
            summary_similarity_matrix = self._calculate_summary_similarity(summaries)

            # Group articles based on both keyword matches and title similarity
            processed_ids = set()
            total_processed = 0

            with db.atomic():
                for i, article in enumerate(ungrouped_articles):
                    if article.id in processed_ids:
                        continue

                    # First, check if the article matches any existing group
                    matched_group = None
                    matched_article = None
                    match_details = None

                    # Check against existing groups
                    for group in existing_groups:
                        articles_in_group = list(News.select().where(News.group == group))

                        # Check if the new article matches any article in the group
                        for group_article in articles_in_group:
                            is_related, details = self._are_articles_related(
                                article,
                                group_article,
                                title_similarity_matrix,
                                summary_similarity_matrix,
                                (i, i)
                            )
                            if is_related:
                                matched_group = group
                                matched_article = group_article
                                match_details = details
                                break

                        if matched_group:
                            break

                    if matched_group:
                        # Add article to existing group
                        all_articles = list(News.select().where(News.group == matched_group))
                        all_articles.append(article)

                        # Update group title if new article has a shorter title
                        shortest_title = self._get_shortest_title(all_articles)
                        if len(shortest_title) < len(matched_group.title):
                            matched_group.title = shortest_title

                        # Update group keywords
                        all_keywords = set()
                        for art in all_articles:
                            art_keywords = self._normalize_keywords(self._parse_keywords(art.keywords))
                            all_keywords.update(art_keywords)
                        matched_group.keywords = json.dumps(list(all_keywords))

                        # Update group date if needed
                        earliest_date = self._get_earliest_date(all_articles)
                        if earliest_date < matched_group.date:
                            matched_group.date = earliest_date

                        # Save updates
                        matched_group.save()
                        article.group = matched_group
                        article.save()

                        processed_ids.add(article.id)
                        total_processed += 1

                        # Print match details
                        print(f"\nAdded article to existing group '{matched_group.title}'")
                        print(f"- New article: {article.title}")
                        print(f"- Matched with: {matched_article.title}")
                        print(f"- Match details:")
                        print(f"  • Exact keyword matches: {match_details['exact_keyword_matches']}/{match_details['min_required_matches']} required")
                        print(f"  • Keyword similarity: {match_details['keyword_similarity']:.3f} (threshold: {self.keyword_similarity_threshold})")
                        print(f"  • Title similarity: {match_details['title_similarity']:.3f} (threshold: {self.title_similarity_threshold})")
                        print(f"  • Summary similarity: {match_details['summary_similarity']:.3f} (threshold: {self.summary_similarity_threshold})")
                        print(f"  • Matched by: {', '.join(match_details['matched_by'])}")
                        continue

                    # If no existing group matched, find related articles in current batch
                    related_indices = []
                    related_match_details = []  # Store match details for each related article
                    for j in range(len(ungrouped_articles)):
                        if (i != j and
                            ungrouped_articles[j].id not in processed_ids):
                            is_related, match_details = self._are_articles_related(
                                article,
                                ungrouped_articles[j],
                                title_similarity_matrix,
                                summary_similarity_matrix,
                                (i, j)
                            )
                            if is_related:
                                related_indices.append(j)
                                related_match_details.append((ungrouped_articles[j], match_details))

                    if not related_indices:
                        # Create single article group if no similar articles found
                        if article.group is None:
                            group = NewsGroup.create(
                                date=article.date,  # Use article's date for single article group
                                title=article.title,
                                summary=f"Single article about: {article.title}",
                                keywords=article.keywords
                            )
                            article.group = group
                            article.save()
                            processed_ids.add(article.id)
                            total_processed += 1
                            print(f"Created single article group: '{article.title}'")
                        continue

                    # Get related articles
                    related_articles = [ungrouped_articles[j] for j in related_indices]
                    all_articles = [article] + related_articles

                    # Get combined keywords for the group
                    all_keywords = set()
                    for art in all_articles:
                        art_keywords = self._normalize_keywords(self._parse_keywords(art.keywords))
                        all_keywords.update(art_keywords)

                    # Create new group
                    group = NewsGroup.create(
                        date=self._get_earliest_date(all_articles),
                        title=self._get_shortest_title(all_articles),
                        summary=f"Group of {len(all_articles)} related articles",
                        keywords=json.dumps(list(all_keywords))
                    )

                    # Update articles to point to the new group
                    article_ids = [art.id for art in all_articles]
                    updated = News.update(group=group).where(News.id.in_(article_ids)).execute()

                    processed_ids.update(article_ids)
                    total_processed += updated

                    # Print detailed matching information
                    print(f"\nCreated new group '{group.title}' with {updated} articles")
                    print(f"- Main article: {article.title}")
                    print(f"- Group date: {group.date}")
                    print(f"- Related articles:")
                    for related_article, match_details in related_match_details:
                        print(f"\n  • {related_article.title}")
                        print(f"    Match details:")
                        print(f"    - Exact keyword matches: {match_details['exact_keyword_matches']}/{match_details['min_required_matches']} required")
                        print(f"    - Keyword similarity: {match_details['keyword_similarity']:.3f} (threshold: {self.keyword_similarity_threshold})")
                        print(f"    - Title similarity: {match_details['title_similarity']:.3f} (threshold: {self.title_similarity_threshold})")
                        print(f"    - Summary similarity: {match_details['summary_similarity']:.3f} (threshold: {self.summary_similarity_threshold})")
                        print(f"    - Matched by: {', '.join(match_details['matched_by'])}")
                    print(f"Combined normalized keywords: {sorted(all_keywords)}")

            return total_processed

        except Exception as e:
            print(f"Error processing articles: {str(e)}")
            traceback.print_exc()
            return 0

    def group_all_ungrouped(self, max_batches: int = None):
        """Process all ungrouped articles in batches.

        Args:
            max_batches: Maximum number of batches to process (None for all)
        """
        batches_processed = 0
        total_articles_processed = 0

        while True:
            articles_processed = self.process_ungrouped_articles()
            if not articles_processed:
                break

            total_articles_processed += articles_processed
            batches_processed += 1
            print(f"Processed batch {batches_processed}, total articles processed: {total_articles_processed}")

            if max_batches and batches_processed >= max_batches:
                break

        return total_articles_processed
