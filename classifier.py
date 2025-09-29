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
                 initial_categories: List[str] = None):
        """
        Initialize the modern text classifier using sentence transformers.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
            similarity_threshold: Minimum similarity to assign to existing category
            initial_categories: List of initial category names
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.categories = set(initial_categories) if initial_categories else set()
        self.category_sentences = defaultdict(list)
        self.category_embeddings = {}  # Store category representative embeddings
        self.sentence_embeddings = {}  # Cache sentence embeddings
        
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
    
    def _find_best_category(self, sentence_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the best matching category using cosine similarity.
        
        Args:
            sentence_embedding: The embedding of the sentence to classify
            
        Returns:
            Tuple of (best_category, similarity_score)
        """
        if not self.category_embeddings:
            return None, 0.0
        
        best_category = None
        best_similarity = 0.0
        
        for category, category_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                sentence_embedding.reshape(1, -1),
                category_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        return best_category, best_similarity
    
    def _generate_category_name(self, sentence: str, embedding: np.ndarray) -> str:
        """
        Generate a meaningful category name using keyword extraction.
        """
        # Simple keyword extraction - get important words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'what', 'about', 'when', 'they', 'some', 'more', 'very', 'into', 'after', 'first', 'well', 'also'}
        meaningful_words = [word for word in words if word not in stop_words]
        
        if meaningful_words:
            # Use the first meaningful word as category name
            return meaningful_words[0]
        else:
            return "miscellaneous"
    
    def classify_sentence(self, sentence: str) -> str:
        """
        Classify a sentence into an appropriate category using transformers.
        
        Args:
            sentence: The text sentence to classify
            
        Returns:
            The category name assigned to the sentence
        """
        if not sentence or not sentence.strip():
            return "empty"
        
        # Get sentence embedding
        sentence_embedding = self._get_embedding(sentence)
        
        # Find best matching category
        best_category, similarity = self._find_best_category(sentence_embedding)
        
        # If similarity is above threshold, use existing category
        if best_category and similarity >= self.similarity_threshold:
            category = best_category
        else:
            # Create new category
            category = self._generate_category_name(sentence, sentence_embedding)
            
            # Ensure category name is unique
            original_category = category
            counter = 1
            while category in self.categories:
                category = f"{original_category}_{counter}"
                counter += 1
            
            self.categories.add(category)
            print(f"New category '{category}' created (similarity: {similarity:.3f})")
        
        # Add sentence to category
        self.category_sentences[category].append(sentence)
        
        # Update category embedding
        self._update_category_embedding(category)
        
        return category
    
    def get_category_similarity(self, sentence: str, category: str) -> float:
        """Get similarity score between a sentence and a category."""
        if category not in self.category_embeddings:
            return 0.0
        
        sentence_embedding = self._get_embedding(sentence)
        category_embedding = self.category_embeddings[category]
        
        return cosine_similarity(
            sentence_embedding.reshape(1, -1),
            category_embedding.reshape(1, -1)
        )[0][0]
    
    def suggest_merge_categories(self, similarity_threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Suggest categories that might be merged based on similarity.
        
        Returns:
            List of tuples (category1, category2, similarity_score)
        """
        suggestions = []
        categories = list(self.categories)
        
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                
                if cat1 in self.category_embeddings and cat2 in self.category_embeddings:
                    similarity = cosine_similarity(
                        self.category_embeddings[cat1].reshape(1, -1),
                        self.category_embeddings[cat2].reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= similarity_threshold:
                        suggestions.append((cat1, cat2, similarity))
        
        return sorted(suggestions, key=lambda x: x[2], reverse=True)
    
    def merge_categories(self, source_category: str, target_category: str):
        """Merge source category into target category."""
        if source_category not in self.categories or target_category not in self.categories:
            raise ValueError("Both categories must exist")
        
        # Move sentences
        self.category_sentences[target_category].extend(self.category_sentences[source_category])
        
        # Remove source category
        del self.category_sentences[source_category]
        self.categories.remove(source_category)
        
        if source_category in self.category_embeddings:
            del self.category_embeddings[source_category]
        
        # Update target category embedding
        self._update_category_embedding(target_category)
        
        print(f"Merged '{source_category}' into '{target_category}'")
    
    def get_categories(self) -> Set[str]:
        """Return all current categories."""
        return self.categories.copy()
    
    def get_sentences_by_category(self, category: str) -> List[str]:
        """Get all sentences in a specific category."""
        return self.category_sentences[category].copy()
    
    def get_all_categorized_sentences(self) -> Dict[str, List[str]]:
        """Get all sentences organized by category."""
        return dict(self.category_sentences)
    
    def save_model(self, filepath: str):
        """Save the classifier state to a file."""
        state = {
            'categories': list(self.categories),
            'category_sentences': dict(self.category_sentences),
            'category_embeddings': {k: v.tolist() for k, v in self.category_embeddings.items()},
            'similarity_threshold': self.similarity_threshold
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load the classifier state from a file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.categories = set(state['categories'])
        self.category_sentences = defaultdict(list, state['category_sentences'])
        self.category_embeddings = {k: np.array(v) for k, v in state['category_embeddings'].items()}
        self.similarity_threshold = state['similarity_threshold']

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