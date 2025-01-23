import json
from typing import List
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc

from .news import News

# Custom spaCy component for TextRank
@Language.component("textrank_component")
def textrank_component(doc: Doc) -> Doc:
    """Custom spaCy component for TextRank"""
    return doc

def get_nlp():
    """Initialize spaCy with custom components and optimized settings."""
    # Load medium model for better accuracy while maintaining good speed
    nlp = spacy.load("en_core_web_md",
                     # Disable components we don't use to improve speed
                     disable=["ner", "entity_ruler", "entity_linker", "textcat", "textcat_multilabel"])

    # Only enable components we need
    nlp.enable_pipe("tagger")      # For POS tagging
    nlp.enable_pipe("parser")      # For dependency parsing
    nlp.enable_pipe("lemmatizer")  # For lemmatization

    # Add our custom TextRank component
    nlp.add_pipe("textrank_component", last=True)

    # Increase max text length for longer articles
    nlp.max_length = 2000000

    return nlp

def process_batch(batch_data: List[dict]) -> List[dict]:
    """Process a batch of articles. This function runs in its own process."""

    # Initialize spaCy
    nlp = get_nlp()

    def extract_keywords(text: str, num_keywords: int = 50) -> List[str]:
        """Extract keywords using spaCy's linguistic features including n-grams."""
        doc = nlp(text)

        # Get important noun phrases and named entities
        keywords = []

        # Extract noun chunks (phrases) with improved scoring
        noun_chunks = list(doc.noun_chunks)
        chunk_scores = []
        for chunk in noun_chunks:
            # Enhanced scoring based on token attributes and position
            score = sum(not token.is_stop and not token.is_punct and
                       token.pos_ in {'NOUN', 'PROPN', 'ADJ'} and
                       (2.0 if token.pos_ in {'PROPN'} else  # Boost proper nouns
                        1.5 if token.dep_ in {'ROOT', 'nsubj', 'dobj'} else  # Boost important dependencies
                        1.0)
                       for token in chunk)
            # Boost score for chunks appearing in title position (first 1/4 of text)
            if chunk.start < len(doc) // 4:
                score *= 1.2
            chunk_scores.append((chunk.text, score))

        # Sort by score and add to keywords
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        keywords.extend(chunk.lower() for chunk, score in chunk_scores[:num_keywords])

        # Extract bigrams and trigrams using dependency parsing with improved scoring
        ngrams = []
        for token in doc:
            # For bigrams: check token and its children
            if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop:
                # Look for adjective + noun combinations
                for child in token.children:
                    if child.pos_ == 'ADJ' and not child.is_stop:
                        bigram = f"{child.text} {token.text}".lower()
                        if bigram not in keywords:
                            # Higher score for adjective-noun pairs with important dependencies
                            score = 2.5 if token.dep_ in {'ROOT', 'nsubj', 'dobj'} else 2.0
                            ngrams.append((bigram, score))

                # Look for noun + noun combinations (compounds)
                for child in token.children:
                    if child.dep_ == 'compound' and not child.is_stop:
                        bigram = f"{child.text} {token.text}".lower()
                        if bigram not in keywords:
                            # Higher score for compound nouns with important dependencies
                            score = 2.2 if token.dep_ in {'ROOT', 'nsubj', 'dobj'} else 1.8
                            ngrams.append((bigram, score))

            # For trigrams: check token and its dependencies with improved pattern matching
            if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop:
                children = list(token.children)
                for i, child1 in enumerate(children):
                    if child1.pos_ == 'ADJ' and not child1.is_stop:
                        # Look for patterns like "ADJ + ADJ + NOUN" or "ADJ + NOUN + NOUN"
                        for child2 in children[i+1:]:
                            if ((child2.pos_ == 'ADJ' and not child2.is_stop) or
                                (child2.dep_ == 'compound' and not child2.is_stop)):
                                trigram = f"{child1.text} {child2.text} {token.text}".lower()
                                if trigram not in keywords:
                                    # Higher score for meaningful trigrams with important dependencies
                                    score = 3.0 if token.dep_ in {'ROOT', 'nsubj', 'dobj'} else 2.5
                                    ngrams.append((trigram, score))

        # Sort n-grams by score and add to keywords
        ngrams.sort(key=lambda x: x[1], reverse=True)
        keywords.extend(ngram for ngram, score in ngrams[:num_keywords])

        # Add important single tokens with enhanced frequency weighting
        token_scores = []
        word_freq = {}
        # Calculate word frequencies
        for token in doc:
            if not token.is_stop and not token.is_punct and len(token.text) > 2:
                word_freq[token.text.lower()] = word_freq.get(token.text.lower(), 0) + 1

        for token in doc:
            if (token.pos_ in {'NOUN', 'PROPN', 'ADJ'} and
                not token.is_stop and
                len(token.text) > 2 and
                token.text.lower() not in {k.lower() for k in keywords}):
                # Enhanced scoring based on frequency, position, and dependency
                freq = word_freq.get(token.text.lower(), 0)
                # Boost score for tokens that are roots or have important dependency labels
                dep_boost = 2.0 if token.dep_ == 'ROOT' else 1.5 if token.dep_ in {'nsubj', 'dobj'} else 1.0
                # Boost score for proper nouns
                pos_boost = 1.5 if token.pos_ == 'PROPN' else 1.0
                # Position boost for tokens appearing early in the text
                pos_boost *= 1.2 if token.i < len(doc) // 4 else 1.0
                score = (freq / len(doc)) * dep_boost * pos_boost
                token_scores.append((token.text, score))

        # Sort and add top tokens
        token_scores.sort(key=lambda x: x[1], reverse=True)
        keywords.extend(token.lower() for token, score in token_scores[:num_keywords])

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)

        return unique_keywords[:num_keywords]

    def textrank_summarize(text: str, num_sentences: int = 5) -> str:
        """Generate summary using spaCy's linguistic features."""
        doc = nlp(text)

        # Clean and get sentences with better filtering
        sentences = []
        for sent in doc.sents:
            # Only include sentences that:
            # 1. Have more than 3 words
            # 2. Have at least one noun or proper noun
            # 3. Have at least one verb
            has_noun = any(token.pos_ in {'NOUN', 'PROPN'} for token in sent)
            has_verb = any(token.pos_ == 'VERB' for token in sent)
            if len(sent.text.split()) > 3 and has_noun and has_verb:
                sentences.append(sent.text.strip())

        if len(sentences) <= num_sentences:
            return text

        # Create similarity matrix based on dependency parsing and semantic similarity
        size = len(sentences)
        similarity_matrix = np.zeros((size, size))

        for i in range(size):
            doc_i = nlp(sentences[i])
            # Get main verbs and nouns from sentence i
            main_tokens_i = [token for token in doc_i if token.pos_ in {'NOUN', 'PROPN', 'VERB'} and not token.is_stop]

            for j in range(size):
                if i != j:
                    doc_j = nlp(sentences[j])
                    # Get main verbs and nouns from sentence j
                    main_tokens_j = [token for token in doc_j if token.pos_ in {'NOUN', 'PROPN', 'VERB'} and not token.is_stop]

                    # Calculate token-level similarities
                    similarity_scores = []
                    for token_i in main_tokens_i:
                        for token_j in main_tokens_j:
                            # Consider lemma matches and dependency relations
                            if token_i.lemma_ == token_j.lemma_:
                                similarity_scores.append(1.0)
                            elif token_i.head.lemma_ == token_j.lemma_ or token_j.head.lemma_ == token_i.lemma_:
                                similarity_scores.append(0.8)
                            elif token_i.dep_ == token_j.dep_:
                                similarity_scores.append(0.6)

                    # Average similarity score for the sentence pair
                    if similarity_scores:
                        similarity_matrix[i][j] = sum(similarity_scores) / len(similarity_scores)

        # Normalize similarity matrix
        norm = similarity_matrix.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        similarity_matrix = similarity_matrix / norm

        # Calculate sentence scores using PageRank
        scores = np.ones(size) / size
        damping = 0.85
        epsilon = 1e-8
        max_iter = 100

        # Power iteration
        for _ in range(max_iter):
            prev_scores = scores.copy()
            for i in range(size):
                score_sum = sum(similarity_matrix[i][j] * scores[j]
                              for j in range(size) if i != j)
                scores[i] = (1 - damping) + damping * score_sum

            if np.sum(np.abs(prev_scores - scores)) < epsilon:
                break

        # Get top sentences while maintaining order
        # Add position bias - favor sentences at the start and end of the document
        position_boost = np.zeros(size)
        first_quarter = size // 4
        last_quarter = size - first_quarter
        position_boost[:first_quarter] = 0.2  # Boost first quarter
        position_boost[-first_quarter:] = 0.1  # Boost last quarter
        scores = scores + position_boost

        top_indices = np.argsort(scores)[-num_sentences:]
        summary_sentences = [sentences[i] for i in sorted(top_indices)]

        return ' '.join(summary_sentences)

    def process_single_article(article_data: dict) -> dict:
        """Process a single article using spaCy."""
        try:
            if not article_data['content'].strip():
                print(f"Article {article_data['id']} has no content.")
                return article_data

            # Extract keywords using spaCy
            text = f"{article_data['title']}\n\n{article_data['content']}"
            new_keywords = extract_keywords(text)
            existing_keywords = json.loads(article_data['keywords'])
            keywords = list(set(new_keywords + existing_keywords))
            article_data['keywords'] = json.dumps(keywords)

            # Generate summary using spaCy
            article_data['summary'] = textrank_summarize(article_data['content'])
            article_data['processed'] = True

            print(f"Processed article: {article_data['title']}")
            return article_data

        except Exception as e:
            print(f"Error processing article {article_data['id']}: {str(e)}")
            return article_data

    processed_batch = []
    # Process articles in the batch using threading
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_single_article, article_data): article_data
            for article_data in batch_data
        }
        for future in futures:
            try:
                processed_data = future.result()
                processed_batch.append(processed_data)
            except Exception as e:
                print(f"Error in thread processing: {str(e)}")
                processed_batch.append(futures[future])

    return processed_batch

class NewsProcessor:
    @staticmethod
    def is_summary_article(title: str) -> bool:
        """Check if an article appears to be a summary or consolidation of multiple articles."""
        summary_indicators = [
            'roundup', 'round-up', 'round up',
            'wrap', 'wrap-up',
            'digest',
            'recap',
            'summary',
            'what you need to know',
            'morning brief',
            'evening brief',
            'key points',
            'key moments',
            'highlights',
            'live updates',
            'live blog',
            'as it happened'
        ]

        # Patterns that indicate multi-topic or comparative articles
        comparative_indicators = [
            ' vs ', ' vs: ', ' versus ',  # Direct comparisons
            ': how ', ' as ',  # Relationship indicators
            ' and ', ' & ',    # Multiple topic connectors
            ' amid ',          # Context connectors
            ' while ',        # Simultaneous events
        ]

        lower_title = title.lower()

        # Check for summary indicators
        if any(indicator in lower_title for indicator in summary_indicators):
            return True

        # Check for comparative patterns
        if any(indicator in title for indicator in comparative_indicators):
            # Additional check to avoid false positives
            # Count significant words on both sides of the comparative indicator
            for indicator in comparative_indicators:
                if indicator in title:
                    parts = title.split(indicator)
                    if len(parts) == 2:
                        # Count significant words (length > 2) on each side
                        words_before = len([w for w in parts[0].split() if len(w) > 2])
                        words_after = len([w for w in parts[1].split() if len(w) > 2])
                        # If both sides have significant content, likely a comparative article
                        if words_before >= 2 and words_after >= 2:
                            return True

        return False

    @classmethod
    def process_all_unprocessed(cls):
        """Process all unprocessed articles using multiprocessing."""
        try:
            unprocessed = News.select().where(News.processed == False)
            total = len(unprocessed)

            if total == 0:
                print("No unprocessed articles found")
                return

            print(f"\n\nProcessing {total} unprocessed articles\n\n")

            # Convert to list of dicts for processing
            articles_data = [{
                'id': article.id,
                'title': article.title,
                'content': article.content,
                'keywords': article.keywords,
                'summary': article.summary,
                'processed': article.processed,
                'is_summary': cls.is_summary_article(article.title)  # Add this flag
            } for article in unprocessed]

            # Process in batches
            batch_size = 30

            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=5) as executor:
                for i in range(0, total, batch_size):
                    batch = articles_data[i:i + batch_size]

                    # Process batch
                    future = executor.submit(process_batch, batch)
                    try:
                        processed_batch = future.result()
                        # Update database with results
                        for processed_data in processed_batch:
                            if processed_data['is_summary']:
                                # Delete summary articles
                                News.delete().where(News.id == processed_data['id']).execute()
                                print(f"Deleted summary article: {processed_data['title']}")
                            else:
                                News.update(
                                    keywords=processed_data['keywords'],
                                    summary=processed_data['summary'],
                                    processed=processed_data['processed']
                                ).where(News.id == processed_data['id']).execute()
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")

                gc.collect()

        finally:
            gc.collect()
