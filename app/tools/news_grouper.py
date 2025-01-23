from datetime import date
import json
import traceback
from typing import List
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.language import Language

from .news import News, NewsGroup

@Language.component("similarity_component")
def similarity_component(doc: Doc) -> Doc:
    """Custom spaCy component optimized for news grouping."""
    # Cache only what we need for grouping
    doc.user_data["main_tokens"] = [
        token for token in doc
        if (token.pos_ in {'NOUN', 'PROPN'} and  # Focus on nouns for grouping
            not token.is_stop and
            len(token.text.strip()) > 2)  # Filter out short tokens
    ]
    # Cache lemmas for quick matching
    doc.user_data["lemmas"] = {
        token.lemma_ for token in doc.user_data["main_tokens"]
    }
    return doc

def get_nlp():
    """Initialize spaCy with optimized settings for news grouping."""
    # Use small model since we only need basic linguistic features
    nlp = spacy.load("en_core_web_sm",
                     # Disable unnecessary components
                     disable=["ner", "entity_ruler", "entity_linker", "textcat", "textcat_multilabel"])

    # Enable only essential components for grouping
    nlp.enable_pipe("tagger")      # For POS tagging
    nlp.enable_pipe("lemmatizer")  # For lemmatization
    nlp.enable_pipe("parser")      # For basic dependency parsing

    # Add custom similarity component
    nlp.add_pipe("similarity_component", last=True)

    # Increase max text length but keep it reasonable for grouping
    nlp.max_length = 1000000  # 1M chars should be enough for grouping

    return nlp

class KeywordBasedNewsGrouper:
    def __init__(self,
                 min_keyword_match_ratio: float = 0.35,
                 keyword_similarity_threshold: float = 0.45,
                 title_similarity_threshold: float = 0.45,
                 summary_similarity_threshold: float = 0.45,
                 content_similarity_threshold: float = 0.55,
                 debug: bool = False):
        """Initialize the KeywordBasedNewsGrouper with more lenient thresholds."""
        self.min_keyword_match_ratio = min_keyword_match_ratio
        self.keyword_similarity_threshold = keyword_similarity_threshold
        self.title_similarity_threshold = title_similarity_threshold
        self.summary_similarity_threshold = summary_similarity_threshold
        self.content_similarity_threshold = content_similarity_threshold
        self.debug = debug

        # Initialize spaCy with batch processing
        self.nlp = get_nlp()
        self.nlp.max_length = 2000000  # Increase max text length

        # Cache for processed documents
        self._doc_cache = {}

    def _get_doc(self, text: str) -> Doc:
        """Get cached spaCy doc or process new one with better error handling."""
        try:
            if not isinstance(text, str):
                print(f"Warning: Invalid input type {type(text)}, expected string")
                return None

            # Clean and normalize the text
            text = text.strip()
            if not text or len(text) < 3:  # Ignore very short texts
                return None

            # Check cache first
            if text in self._doc_cache:
                return self._doc_cache[text]

            # Process new document
            doc = self.nlp(text)

            # Ensure the document was processed successfully
            if not doc or not hasattr(doc, 'user_data'):
                print(f"Warning: Failed to process text properly: {text[:50]}...")
                return None

            # Initialize user_data if not present
            if 'main_tokens' not in doc.user_data:
                doc.user_data["main_tokens"] = [token for token in doc
                                              if token.pos_ in {'NOUN', 'PROPN', 'VERB'}
                                              and not token.is_stop]
            if 'lemmas' not in doc.user_data:
                doc.user_data["lemmas"] = {token.lemma_ for token in doc.user_data["main_tokens"]}

            # Cache the document
            self._doc_cache[text] = doc

            # Manage cache size
            if len(self._doc_cache) > 1000:
                # Remove oldest entries
                old_keys = list(self._doc_cache.keys())[:-500]  # Keep last 500 entries
                for k in old_keys:
                    del self._doc_cache[k]

            return doc

        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {str(e)}")
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate optimized similarity between two texts using spaCy."""
        try:
            if not text1 or not text2:
                return 0.0

            # Get cached or process docs
            doc1 = self._get_doc(text1)
            doc2 = self._get_doc(text2)

            # Handle cases where document processing failed
            if not doc1 or not doc2:
                return 0.0

            # Ensure user_data is present
            if not hasattr(doc1, 'user_data') or not hasattr(doc2, 'user_data'):
                return 0.0

            # Get cached main tokens and lemmas with safety checks
            main_tokens1 = doc1.user_data.get("main_tokens", [])
            main_tokens2 = doc2.user_data.get("main_tokens", [])
            lemmas1 = doc1.user_data.get("lemmas", set())
            lemmas2 = doc2.user_data.get("lemmas", set())

            if not main_tokens1 or not main_tokens2 or not lemmas1 or not lemmas2:
                return 0.0

            # Quick lexical similarity using lemma overlap
            lemma_overlap = len(lemmas1 & lemmas2) / max(len(lemmas1), len(lemmas2))
            if lemma_overlap < 0.1:  # Early exit if very different
                return 0.0

            # Calculate token-level similarities
            similarity_scores = []
            for token1 in main_tokens1:
                if not hasattr(token1, 'lemma_') or not hasattr(token1, 'head'):
                    continue
                for token2 in main_tokens2:
                    if not hasattr(token2, 'lemma_') or not hasattr(token2, 'head'):
                        continue
                    # Consider lemma matches and dependency relations
                    if token1.lemma_ == token2.lemma_:
                        similarity_scores.append(1.0)
                    elif (hasattr(token1.head, 'lemma_') and
                          hasattr(token2.head, 'lemma_') and
                          (token1.head.lemma_ == token2.lemma_ or
                           token2.head.lemma_ == token1.lemma_)):
                        similarity_scores.append(0.8)
                    elif hasattr(token1, 'dep_') and hasattr(token2, 'dep_') and token1.dep_ == token2.dep_:
                        similarity_scores.append(0.6)

            # Calculate final similarity score
            if similarity_scores:
                return sum(similarity_scores) / len(similarity_scores)
            return lemma_overlap  # Fallback to lemma overlap

        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        """Normalize keywords using cached spaCy docs."""
        normalized = []
        for keyword in keywords:
            if not keyword or not isinstance(keyword, str):
                continue
            doc = self._get_doc(keyword)
            if doc is None:  # Skip if document processing failed
                continue
            # Only process valid tokens with lemma attribute
            tokens = [token.lemma_ for token in doc if hasattr(token, 'lemma_')]
            if tokens:  # Only add if we have valid tokens
                normalized.append(' '.join(tokens))
        return normalized if normalized else keywords  # Fallback to original keywords if normalization failed

    def _calculate_keyword_similarity(self, keywords1: List[str], keywords2: List[str]) -> tuple[int, float]:
        """Calculate optimized similarity between two sets of keywords."""
        if not keywords1 or not keywords2:
            return 0, 0.0

        # Process all keywords at once for efficiency
        docs1 = [doc for doc in [self._get_doc(k) for k in keywords1] if doc is not None]
        docs2 = [doc for doc in [self._get_doc(k) for k in keywords2] if doc is not None]

        if not docs1 or not docs2:
            return 0, 0.0

        # Get lemma sets for quick matching
        lemma_sets1 = [{token.lemma_ for token in doc.user_data.get("main_tokens", [])} for doc in docs1]
        lemma_sets2 = [{token.lemma_ for token in doc.user_data.get("main_tokens", [])} for doc in docs2]

        # Filter out empty sets
        lemma_sets1 = [s for s in lemma_sets1 if s]
        lemma_sets2 = [s for s in lemma_sets2 if s]

        if not lemma_sets1 or not lemma_sets2:
            return 0, 0.0

        # Count exact matches using lemma sets
        exact_matches = 0
        for s1 in lemma_sets1:
            for s2 in lemma_sets2:
                if len(s1) > 0 and len(s2) > 0:
                    overlap_ratio = len(s1 & s2) / max(len(s1), len(s2))
                    if overlap_ratio > 0.8:
                        exact_matches += 1

        # Calculate semantic similarity for non-exact matches
        similarity_scores = []
        for i, doc1 in enumerate(docs1):
            if i >= len(lemma_sets1):  # Skip if index out of range
                continue
            for j, doc2 in enumerate(docs2):
                if j >= len(lemma_sets2):  # Skip if index out of range
                    continue
                if not doc1 or not doc2:
                    continue

                # Only proceed if both lemma sets exist and are non-empty
                if i < len(lemma_sets1) and j < len(lemma_sets2) and lemma_sets1[i] and lemma_sets2[j]:
                    overlap = len(lemma_sets1[i] & lemma_sets2[j]) / max(len(lemma_sets1[i]), len(lemma_sets2[j]))
                    if overlap < 0.8:  # Only calculate detailed similarity for non-exact matches
                        main_tokens1 = doc1.user_data.get("main_tokens", [])
                        main_tokens2 = doc2.user_data.get("main_tokens", [])
                        if main_tokens1 and main_tokens2:  # Only proceed if both token lists are non-empty
                            token_sims = []
                            for t1 in main_tokens1:
                                for t2 in main_tokens2:
                                    if t1.lemma_ == t2.lemma_:
                                        token_sims.append(1.0)
                                    elif t1.head.lemma_ == t2.lemma_ or t2.head.lemma_ == t1.lemma_:
                                        token_sims.append(0.8)
                            if token_sims:
                                similarity_scores.append(sum(token_sims) / len(token_sims))

        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        return exact_matches, avg_similarity

    def _are_articles_related(self, article1: News, article2: News) -> tuple[bool, dict]:
        """Determine if two articles are related using spaCy similarity."""
        # Quick check for completely unrelated titles
        title_similarity = self._calculate_similarity(article1.title, article2.title)
        if title_similarity < self.title_similarity_threshold * 0.3:  # Reduced from 0.5
            return False, {}

        # Calculate summary similarity
        summary_similarity = self._calculate_similarity(article1.summary or "", article2.summary or "")

        # Calculate content similarity with more context
        content_similarity = self._calculate_similarity(
            f"{article1.title} {article1.summary or ''} {article1.content[:500]}",  # Include more content
            f"{article2.title} {article2.summary or ''} {article2.content[:500]}"
        )

        # Get keywords
        keywords1 = self._parse_keywords(article1.keywords)
        keywords2 = self._parse_keywords(article2.keywords)
        exact_matches, keyword_similarity = self._calculate_keyword_similarity(keywords1, keywords2)

        # Calculate minimum required matches
        min_keywords = min(len(keywords1), len(keywords2))
        min_required_matches = round(min_keywords * self.min_keyword_match_ratio)

        # Calculate matching score with adjusted weights
        matching_score = 0.0
        matched_by = []

        # Title similarity contribution (increased weight for partial matches)
        if title_similarity >= self.title_similarity_threshold:
            matching_score += 0.4
            matched_by.append('title')
        elif title_similarity >= self.title_similarity_threshold * 0.7:  # More lenient partial match
            matching_score += 0.3

        # Content similarity contribution (increased weight)
        if content_similarity >= self.content_similarity_threshold:
            matching_score += 0.5  # Increased from 0.4
            matched_by.append('content')
        elif content_similarity >= self.content_similarity_threshold * 0.8:
            matching_score += 0.3  # Increased partial match score

        # Keyword similarity contribution
        if keyword_similarity >= self.keyword_similarity_threshold:
            matching_score += 0.5
            matched_by.append('keywords')
        elif keyword_similarity >= self.keyword_similarity_threshold * 0.8:
            matching_score += 0.3

        # Summary similarity contribution
        if summary_similarity >= self.summary_similarity_threshold:
            matching_score += 0.2
            matched_by.append('summary')
        elif summary_similarity >= self.summary_similarity_threshold * 0.8:
            matching_score += 0.1

        # Articles are related if matching score is high enough and we have sufficient keyword matches
        is_related = matching_score >= 0.8 and exact_matches >= min_required_matches  # Reduced from 1.0

        match_details = {
            'title_similarity': title_similarity,
            'summary_similarity': summary_similarity,
            'content_similarity': content_similarity,
            'keyword_similarity': keyword_similarity,
            'exact_keyword_matches': exact_matches,
            'min_required_matches': min_required_matches,
            'matching_score': matching_score,
            'matched_by': matched_by
        }

        return is_related, match_details

    def _get_earliest_date(self, articles: List[News]) -> date:
        """Get the earliest date from a list of articles."""
        return min(article.date for article in articles)

    def _get_shortest_title(self, articles: List[News]) -> str:
        """Get the shortest title from a list of articles that's not empty or None."""
        valid_titles = [article.title for article in articles if article.title and article.title.strip()]
        if not valid_titles:
            return ""
        return min(valid_titles, key=len)

    def _parse_keywords(self, keywords_json: str) -> List[str]:
        """Parse keywords from JSON string."""
        try:
            return json.loads(keywords_json)
        except:
            return []

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
                                    .where(News.processed == True)
                                    .order_by(News.id.desc())
                                    .limit(batch_size))

            if not ungrouped_articles:
                print("No ungrouped articles found")
                return 0

            print(f"\nProcessing batch of {len(ungrouped_articles)} articles")

            # Group articles based on both keyword matches and title similarity
            processed_ids = set()
            total_processed = 0

            for i, article in enumerate(ungrouped_articles):
                if article.id in processed_ids:
                    continue

                # Initialize matching variables
                matched_group = None
                match_details = None

                # Get all existing groups for comparison
                existing_groups = list(NewsGroup.select().order_by(NewsGroup.id.desc()))

                # Check against existing groups
                for group in existing_groups:
                    articles_in_group = list(News.select().where(News.group == group))

                    for group_article in articles_in_group:
                        is_related, match_details = self._are_articles_related(
                            article,
                            group_article
                        )

                        if is_related:
                            matched_group = group
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
                    print()
                    print(f"- New article: {article.title}")
                    print(f"- Matched with: {matched_group.title}")
                    print(f"- Match details:")
                    print(f"  • Exact keyword matches: {match_details['exact_keyword_matches']}/{match_details['min_required_matches']} required")
                    print(f"  • Keyword similarity: {match_details['keyword_similarity']:.3f} (threshold: {self.keyword_similarity_threshold})")
                    print(f"  • Title similarity: {match_details['title_similarity']:.3f} (threshold: {self.title_similarity_threshold})")
                    print(f"  • Summary similarity: {match_details['summary_similarity']:.3f} (threshold: {self.summary_similarity_threshold})")
                    print(f"  • Content similarity: {match_details['content_similarity']:.3f} (threshold: {self.content_similarity_threshold})")
                    print(f"  • Final matching score: {match_details['matching_score']:.3f}")
                    print(f"  • Matched by: {', '.join(match_details['matched_by'])}")
                    print()
                    continue

                # If no existing group matched, find related articles in current batch
                related_indices = []
                related_match_details = []  # Store match details for each related article
                for j in range(len(ungrouped_articles)):
                    if (i != j and
                        ungrouped_articles[j].id not in processed_ids):
                        is_related, match_details = self._are_articles_related(
                            article,
                            ungrouped_articles[j]
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
                    print(f"    - Content similarity: {match_details['content_similarity']:.3f} (threshold: {self.content_similarity_threshold})")
                    print(f"    - Final matching score: {match_details['matching_score']:.3f}")
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
