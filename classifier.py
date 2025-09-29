import re
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch

class ModernTextClassifier:
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.7,
                 initial_categories: List[str] = None,
                 merge_threshold: float = 0.8):
        """
        Initialize the modern text classifier using sentence transformers.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
            similarity_threshold: Minimum similarity to assign to existing category
            initial_categories: List of initial category names
            merge_threshold: Threshold for automatically merging similar categories
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.categories = set(initial_categories) if initial_categories else set()
        self.category_sentences = defaultdict(list)
        self.category_embeddings = {}  # Store category representative embeddings
        self.sentence_embeddings = {}  # Cache sentence embeddings
        self.category_creation_history = []  # Track all categories created
        self.category_keywords = defaultdict(set)  # Track keywords for each category
        self.merge_history = []  # Track category merges
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding with caching."""
        if text not in self.sentence_embeddings:
            self.sentence_embeddings[text] = self.model.encode([text])[0]
        return self.sentence_embeddings[text]
    
    def _update_category_embedding(self, category: str):
        """Update the representative embedding for a category."""
        if category in self.category_sentences and self.category_sentences[category]:
            sentences = self.category_sentences[category]
            embeddings = [self._get_embedding(sent) for sent in sentences]
            # Use mean embedding as category representative
            self.category_embeddings[category] = np.mean(embeddings, axis=0)
    
    def _extract_keywords(self, sentence: str) -> Set[str]:
        """Extract meaningful keywords from a sentence."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        stop_words = {'this', 'that', 'with', 'have', 'will', 'been', 'were', 'said', 
                     'each', 'which', 'their', 'time', 'would', 'there', 'what', 
                     'about', 'when', 'they', 'some', 'more', 'very', 'into', 
                     'after', 'first', 'well', 'also', 'and', 'the', 'for', 'are'}
        return set(word for word in words if word not in stop_words)
    
    def _find_best_category_comprehensive(self, sentence: str, sentence_embedding: np.ndarray) -> Tuple[Optional[str], float, str]:
        """
        Comprehensive category matching using both embeddings and keywords.
        
        Returns:
            Tuple of (best_category, similarity_score, reason)
        """
        if not self.category_embeddings:
            return None, 0.0, "no_existing_categories"
        
        best_category = None
        best_similarity = 0.0
        match_reason = "embedding_similarity"
        
        sentence_keywords = self._extract_keywords(sentence)
        
        # Check embedding similarity
        for category, category_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                sentence_embedding.reshape(1, -1),
                category_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        # Also check keyword overlap for better matching
        keyword_matches = {}
        for category in self.categories:
            if category in self.category_keywords:
                common_keywords = sentence_keywords.intersection(self.category_keywords[category])
                if common_keywords:
                    keyword_score = len(common_keywords) / len(sentence_keywords.union(self.category_keywords[category]))
                    keyword_matches[category] = (keyword_score, common_keywords)
        
        # If we have strong keyword matches, consider them
        if keyword_matches:
            best_keyword_category = max(keyword_matches.keys(), key=lambda k: keyword_matches[k][0])
            keyword_score, common_words = keyword_matches[best_keyword_category]
            
            # If keyword similarity is strong and embedding similarity is weak, prefer keyword match
            if keyword_score > 0.3 and best_similarity < self.similarity_threshold:
                best_category = best_keyword_category
                best_similarity = keyword_score
                match_reason = f"keyword_overlap: {common_words}"
        
        return best_category, best_similarity, match_reason
    
    def _find_similar_category_names(self, proposed_name: str) -> List[Tuple[str, float]]:
        """Find existing categories with similar names."""
        similar_categories = []
        proposed_keywords = self._extract_keywords(proposed_name)
        
        for existing_category in self.categories:
            existing_keywords = self._extract_keywords(existing_category)
            
            # Calculate Jaccard similarity for keywords
            if proposed_keywords or existing_keywords:
                intersection = len(proposed_keywords.intersection(existing_keywords))
                union = len(proposed_keywords.union(existing_keywords))
                jaccard_similarity = intersection / union if union > 0 else 0
                
                if jaccard_similarity > 0.3:  # Threshold for similar names
                    similar_categories.append((existing_category, jaccard_similarity))
        
        return sorted(similar_categories, key=lambda x: x[1], reverse=True)
    
    def _generate_category_name(self, sentence: str, embedding: np.ndarray) -> str:
        """
        Generate a meaningful category name with cross-verification.
        """
        sentence_keywords = self._extract_keywords(sentence)
        
        # Get the most meaningful words (longer words first, then by frequency)
        meaningful_words = sorted(list(sentence_keywords), key=lambda x: (-len(x), x))
        
        if meaningful_words:
            proposed_name = meaningful_words[0]
            
            # Check if similar category names already exist
            similar_categories = self._find_similar_category_names(proposed_name)
            
            if similar_categories:
                print(f"  Warning: Proposed name '{proposed_name}' is similar to existing categories:")
                for cat_name, similarity in similar_categories[:3]:
                    print(f"    - '{cat_name}' (similarity: {similarity:.3f})")
                
                # Use the most similar existing category name as base
                base_name = similar_categories[0][0]
                return base_name
            
            return proposed_name
        else:
            return "miscellaneous"
    
    def display_category_summary(self):
        """Display a summary of all current categories."""
        print(f"\n=== CURRENT CATEGORIES ({len(self.categories)}) ===")
        for i, category in enumerate(sorted(self.categories), 1):
            sentence_count = len(self.category_sentences[category])
            keywords = list(self.category_keywords[category])[:5]  # Show first 5 keywords
            print(f"{i:2d}. {category:<20} ({sentence_count} sentences) Keywords: {keywords}")
        print("=" * 50)
    
    def _merge_categories(self, source_category: str, target_category: str) -> str:
        """
        Merge source category into target category.
        
        Args:
            source_category: Category to be merged (will be removed)
            target_category: Category to merge into (will remain)
            
        Returns:
            The target category name
        """
        if source_category not in self.categories or target_category not in self.categories:
            return target_category
        
        # Move all sentences from source to target
        self.category_sentences[target_category].extend(self.category_sentences[source_category])
        
        # Merge keywords
        self.category_keywords[target_category].update(self.category_keywords[source_category])
        
        # Remove source category
        self.categories.remove(source_category)
        del self.category_sentences[source_category]
        del self.category_keywords[source_category]
        if source_category in self.category_embeddings:
            del self.category_embeddings[source_category]
        
        # Update target category embedding
        self._update_category_embedding(target_category)
        
        # Record the merge
        self.merge_history.append({
            'source': source_category,
            'target': target_category,
            'timestamp': len(self.merge_history) + 1
        })
        
        print(f"  -> MERGED category '{source_category}' into '{target_category}'")
        return target_category
    
    def _check_for_merges(self, current_category: str) -> str:
        """
        Check if current category should be merged with existing categories.
        
        Args:
            current_category: The category to check for potential merges
            
        Returns:
            The final category name (may be different if merged)
        """
        if current_category not in self.category_embeddings:
            return current_category
        
        current_embedding = self.category_embeddings[current_category]
        current_keywords = self.category_keywords[current_category]
        
        best_merge_candidate = None
        best_merge_score = 0.0
        
        for other_category in self.categories:
            if other_category == current_category:
                continue
                
            if other_category not in self.category_embeddings:
                continue
            
            # Calculate embedding similarity
            other_embedding = self.category_embeddings[other_category]
            embedding_similarity = cosine_similarity(
                current_embedding.reshape(1, -1),
                other_embedding.reshape(1, -1)
            )[0][0]
            
            # Calculate keyword similarity
            other_keywords = self.category_keywords[other_category]
            if current_keywords and other_keywords:
                intersection = len(current_keywords.intersection(other_keywords))
                union = len(current_keywords.union(other_keywords))
                keyword_similarity = intersection / union if union > 0 else 0
            else:
                keyword_similarity = 0
            
            # Combined similarity score (weighted average)
            combined_score = 0.7 * embedding_similarity + 0.3 * keyword_similarity
            
            if combined_score > best_merge_score and combined_score >= self.merge_threshold:
                best_merge_score = combined_score
                best_merge_candidate = other_category
        
        # Perform merge if candidate found
        if best_merge_candidate:
            print(f"  -> Auto-merge detected: '{current_category}' + '{best_merge_candidate}' (score: {best_merge_score:.3f})")
            
            # Merge into the category with more sentences (more established)
            current_count = len(self.category_sentences[current_category])
            candidate_count = len(self.category_sentences[best_merge_candidate])
            
            if current_count >= candidate_count:
                return self._merge_categories(best_merge_candidate, current_category)
            else:
                return self._merge_categories(current_category, best_merge_candidate)
        
        return current_category
    
    def _suggest_broader_category_name(self, cat1: str, cat2: str) -> str:
        """
        Suggest a broader category name when merging two categories.
        
        Args:
            cat1: First category name
            cat2: Second category name
            
        Returns:
            Suggested broader category name
        """
        # Extract keywords from both categories
        keywords1 = self._extract_keywords(cat1)
        keywords2 = self._extract_keywords(cat2)
        
        # Find common keywords
        common_keywords = keywords1.intersection(keywords2)
        
        if common_keywords:
            # Use the longest common keyword
            return max(common_keywords, key=len)
        
        # If no common keywords, try to find a broader term
        all_keywords = keywords1.union(keywords2)
        
        # Simple heuristics for broader categories
        if any(word in all_keywords for word in ['sports', 'game', 'play', 'match']):
            return 'sports'
        elif any(word in all_keywords for word in ['food', 'cook', 'recipe', 'eat']):
            return 'food'
        elif any(word in all_keywords for word in ['travel', 'trip', 'vacation', 'journey']):
            return 'travel'
        elif any(word in all_keywords for word in ['technology', 'computer', 'programming', 'software']):
            return 'technology'
        elif any(word in all_keywords for word in ['health', 'medical', 'doctor', 'medicine']):
            return 'health'
        elif any(word in all_keywords for word in ['entertainment', 'movie', 'music', 'show']):
            return 'entertainment'
        else:
            # Return the shorter category name as it's likely more general
            return cat1 if len(cat1) <= len(cat2) else cat2

    def classify_sentence(self, sentence: str) -> str:
        """
        Classify a sentence into an appropriate category with comprehensive cross-verification and dynamic merging.
        
        Args:
            sentence: The text sentence to classify
            
        Returns:
            The category name assigned to the sentence
        """
        if not sentence or not sentence.strip():
            return "empty"
        
        # Display current categories for reference
        if len(self.categories) > 0:
            print(f"\nCurrent categories: {sorted(list(self.categories))}")
        
        # Get sentence embedding and keywords
        sentence_embedding = self._get_embedding(sentence)
        sentence_keywords = self._extract_keywords(sentence)
        
        # Find best matching category with comprehensive analysis
        best_category, similarity, match_reason = self._find_best_category_comprehensive(
            sentence, sentence_embedding
        )
        
        # Decision logic
        if best_category and similarity >= self.similarity_threshold:
            category = best_category
            print(f"  -> Assigned to existing category '{category}' (score: {similarity:.3f}, reason: {match_reason})")
        else:
            # Create new category with cross-verification
            category = self._generate_category_name(sentence, sentence_embedding)
            
            # Ensure category name is unique
            original_category = category
            counter = 1
            while category in self.categories:
                category = f"{original_category}_{counter}"
                counter += 1
            
            self.categories.add(category)
            self.category_creation_history.append({
                'name': category,
                'first_sentence': sentence,
                'timestamp': len(self.category_creation_history) + 1
            })
            
            print(f"  -> NEW category '{category}' created (best existing similarity: {similarity:.3f})")
            if similarity > 0.3:  # Show if there was a decent match that wasn't quite enough
                print(f"     (closest existing category was '{best_category}')")
        
        # Add sentence to category and update keywords
        self.category_sentences[category].append(sentence)
        self.category_keywords[category].update(sentence_keywords)
        
        # Update category embedding
        self._update_category_embedding(category)
        
        # Check for potential merges after adding the sentence
        final_category = self._check_for_merges(category)
        
        return final_category
    
    def get_merge_history(self) -> List[Dict]:
        """Get the history of category merges."""
        return self.merge_history.copy()
    
    def force_merge_categories(self, cat1: str, cat2: str, new_name: str = None) -> str:
        """
        Manually force merge two categories.
        
        Args:
            cat1: First category to merge
            cat2: Second category to merge  
            new_name: Optional new name for merged category
            
        Returns:
            The final merged category name
        """
        if cat1 not in self.categories or cat2 not in self.categories:
            print(f"Error: One or both categories don't exist: {cat1}, {cat2}")
            return cat1 if cat1 in self.categories else cat2
        
        # Determine target category name
        if new_name:
            target_category = new_name
            # Create new category with merged content
            self.categories.add(target_category)
            self.category_sentences[target_category] = (
                self.category_sentences[cat1] + self.category_sentences[cat2]
            )
            self.category_keywords[target_category] = (
                self.category_keywords[cat1].union(self.category_keywords[cat2])
            )
            
            # Remove old categories
            self.categories.discard(cat1)
            self.categories.discard(cat2)
            del self.category_sentences[cat1]
            del self.category_sentences[cat2]
            del self.category_keywords[cat1]
            del self.category_keywords[cat2]
            if cat1 in self.category_embeddings:
                del self.category_embeddings[cat1]
            if cat2 in self.category_embeddings:
                del self.category_embeddings[cat2]
        else:
            # Use suggested broader name
            suggested_name = self._suggest_broader_category_name(cat1, cat2)
            target_category = suggested_name if suggested_name != cat1 and suggested_name != cat2 else cat1
            target_category = self._merge_categories(cat2, cat1)
        
        # Update embedding for merged category
        self._update_category_embedding(target_category)
        
        return target_category

# Convenience function for simple usage
def classify_text_modern(sentence: str, classifier: ModernTextClassifier = None) -> str:
    """
    Modern function to classify a single sentence using transformers.
    
    Args:
        sentence: The text sentence to classify
        classifier: Optional existing classifier instance
        
    Returns:
        The category assigned to the sentence
    """
    if classifier is None:
        classifier = ModernTextClassifier()
    
    return classifier.classify_sentence(sentence)

# Example usage
if __name__ == "__main__":
    print("Loading transformer model...")
    classifier = ModernTextClassifier(similarity_threshold=0.6)
    
    # Test sentences
    test_sentences = [
        "I love programming in Python and building AI applications",
        "The football match was incredible, our team scored 3 goals",
        "Let's cook a delicious pasta dinner with fresh ingredients",
        "Planning an amazing vacation trip to Paris next summer",
        "Need to schedule a doctor appointment for my annual checkup",
        "Watched an amazing sci-fi movie with great special effects",
        "Studying machine learning algorithms for my computer science exam",
        "Working on an important client project with tight deadlines",
        "Playing soccer with my children in the neighborhood park",
        "Heavy rainfall expected throughout the weekend",
        "Debugging code is challenging but rewarding work",
        "Basketball practice starts at 6 PM today",
        "Recipe for chocolate cake looks delicious",
        "Flight booking confirmed for London vacation"
    ]
    
    print("\nClassifying sentences...")
    for sentence in test_sentences:
        category = classifier.classify_sentence(sentence)
        print(f"'{sentence[:50]}...' -> Category: {category}")
    
    print(f"\nTotal categories created: {len(classifier.get_categories())}")
    print(f"Categories: {sorted(classifier.get_categories())}")
    
    # Show merge suggestions
    print("\nSuggested category merges:")
    merge_suggestions = classifier.suggest_merge_categories(0.7)
    for cat1, cat2, sim in merge_suggestions[:5]:  # Show top 5
        print(f"  {cat1} + {cat2} (similarity: {sim:.3f})")