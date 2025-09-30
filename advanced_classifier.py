import re
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch.nn.functional as F

class AdvancedTextClassifier:
    def __init__(self, 
                 model_name: str = 'microsoft/DialoGPT-medium',  # Better for Q&A understanding
                 similarity_threshold: float = 0.75,
                 initial_categories: List[str] = None,
                 merge_threshold: float = 0.85,
                 device: str = None):
        """
        Initialize the advanced text classifier using transformer models.
        
        Args:
            model_name: HuggingFace model name for embeddings (DialoGPT, BERT-large, RoBERTa, etc.)
            similarity_threshold: Minimum similarity to assign to existing category
            initial_categories: List of initial category names
            merge_threshold: Threshold for automatically merging similar categories
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing classifier on device: {self.device}")
        
        # Initialize more powerful models
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.categories = set(initial_categories) if initial_categories else set()
        self.category_questions = defaultdict(list)  # Store full questions instead of sentences
        self.category_embeddings = {}  # Store category representative embeddings
        self.question_embeddings = {}  # Cache question embeddings
        self.category_creation_history = []  # Track all categories created
        self.category_keywords = defaultdict(set)  # Track keywords for each category
        self.merge_history = []  # Track category merges
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get question embedding using transformer model with caching."""
        if text not in self.question_embeddings:
            # Tokenize with proper padding and attention masks
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling across sequence length
                    attention_mask = inputs['attention_mask']
                    embeddings = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()
                else:
                    # Fallback to pooler output if available
                    embedding = outputs.pooler_output.squeeze().cpu().numpy()
            
            self.question_embeddings[text] = embedding
        
        return self.question_embeddings[text]
    
    def _update_category_embedding(self, category: str):
        """Update the representative embedding for a category."""
        if category in self.category_questions and self.category_questions[category]:
            questions = self.category_questions[category]
            embeddings = [self._get_embedding(question) for question in questions]
            # Use mean embedding as category representative
            self.category_embeddings[category] = np.mean(embeddings, axis=0)
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from a question."""
        # More sophisticated keyword extraction for questions
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Enhanced stop words for Q&A context
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom',
            'this', 'that', 'with', 'have', 'will', 'been', 'were', 'said', 
            'each', 'their', 'time', 'would', 'there', 'about', 'they', 'some', 
            'more', 'very', 'into', 'after', 'first', 'well', 'also', 'and', 
            'the', 'for', 'are', 'you', 'can', 'could', 'should', 'would',
            'does', 'did', 'has', 'had', 'was', 'were', 'been', 'being'
        }
        
        # Extract meaningful words (nouns, verbs, adjectives)
        meaningful_words = set(word for word in words if word not in stop_words and len(word) > 2)
        
        # Also extract potential domain-specific terms
        domain_indicators = re.findall(r'\b(?:programming|python|machine|learning|cooking|travel|health|sports)\b', text.lower())
        meaningful_words.update(domain_indicators)
        
        return meaningful_words
    
    def _analyze_question_intent(self, question: str) -> Dict[str, any]:
        """Analyze the intent and structure of a question."""
        question_lower = question.lower().strip()
        
        intent_patterns = {
            'definition': [r'what is', r'what are', r'define', r'meaning of'],
            'procedure': [r'how to', r'how do', r'how can', r'steps to'],
            'comparison': [r'difference between', r'compare', r'vs', r'versus'],
            'recommendation': [r'best', r'recommend', r'suggest', r'which should'],
            'troubleshooting': [r'error', r'problem', r'issue', r'not working', r'fix'],
            'factual': [r'when did', r'where is', r'who was', r'capital of']
        }
        
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            if any(re.search(pattern, question_lower) for pattern in patterns):
                detected_intents.append(intent)
        
        return {
            'intents': detected_intents,
            'question_type': detected_intents[0] if detected_intents else 'general',
            'length': len(question.split()),
            'complexity': 'high' if len(question.split()) > 15 else 'medium' if len(question.split()) > 8 else 'simple'
        }
    
    def _find_best_category_comprehensive(self, question: str, question_embedding: np.ndarray) -> Tuple[Optional[str], float, str]:
        """
        Comprehensive category matching using embeddings, keywords, and question intent.
        
        Returns:
            Tuple of (best_category, similarity_score, reason)
        """
        if not self.category_embeddings:
            return None, 0.0, "no_existing_categories"
        
        best_category = None
        best_similarity = 0.0
        match_reason = "embedding_similarity"
        
        question_keywords = self._extract_keywords(question)
        question_intent = self._analyze_question_intent(question)
        
        # Check embedding similarity with all categories
        embedding_scores = {}
        for category, category_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                question_embedding.reshape(1, -1),
                category_embedding.reshape(1, -1)
            )[0][0]
            embedding_scores[category] = similarity
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        # Enhanced keyword matching
        keyword_matches = {}
        for category in self.categories:
            if category in self.category_keywords:
                common_keywords = question_keywords.intersection(self.category_keywords[category])
                if common_keywords:
                    # Weighted keyword score based on keyword importance
                    keyword_score = len(common_keywords) / max(len(question_keywords), len(self.category_keywords[category]))
                    keyword_matches[category] = (keyword_score, common_keywords)
        
        # Intent-based matching (check if category questions have similar intents)
        intent_matches = {}
        for category in self.categories:
            if category in self.category_questions:
                similar_intents = 0
                for cat_question in self.category_questions[category]:
                    cat_intent = self._analyze_question_intent(cat_question)
                    if any(intent in cat_intent['intents'] for intent in question_intent['intents']):
                        similar_intents += 1
                
                if similar_intents > 0:
                    intent_score = similar_intents / len(self.category_questions[category])
                    intent_matches[category] = intent_score
        
        # Combined scoring with weights
        final_scores = {}
        for category in self.categories:
            embedding_score = embedding_scores.get(category, 0.0)
            keyword_score = keyword_matches.get(category, (0.0, set()))[0]
            intent_score = intent_matches.get(category, 0.0)
            
            # Weighted combination
            combined_score = (
                0.6 * embedding_score +      # Primary: semantic similarity
                0.25 * keyword_score +       # Secondary: keyword overlap
                0.15 * intent_score          # Tertiary: intent similarity
            )
            
            final_scores[category] = combined_score
        
        # Find best category with combined scoring
        if final_scores:
            best_category = max(final_scores.keys(), key=lambda k: final_scores[k])
            best_similarity = final_scores[best_category]
            
            # Determine match reason
            embedding_contrib = 0.6 * embedding_scores.get(best_category, 0.0)
            keyword_contrib = 0.25 * keyword_matches.get(best_category, (0.0, set()))[0]
            intent_contrib = 0.15 * intent_matches.get(best_category, 0.0)
            
            if embedding_contrib >= keyword_contrib and embedding_contrib >= intent_contrib:
                match_reason = "semantic_similarity"
            elif keyword_contrib >= intent_contrib:
                common_words = keyword_matches.get(best_category, (0.0, set()))[1]
                match_reason = f"keyword_overlap: {list(common_words)[:3]}"
            else:
                match_reason = "intent_similarity"
        
        return best_category, best_similarity, match_reason
    
    def _generate_category_name(self, question: str, embedding: np.ndarray) -> str:
        """Generate a meaningful category name based on question analysis."""
        question_keywords = self._extract_keywords(question)
        question_intent = self._analyze_question_intent(question)
        
        # Priority-based category naming
        # 1. Check for domain-specific keywords
        domain_keywords = {
            'programming': ['python', 'code', 'programming', 'software', 'algorithm', 'debugging'],
            'cooking': ['cook', 'recipe', 'food', 'ingredient', 'kitchen', 'meal'],
            'travel': ['travel', 'trip', 'vacation', 'flight', 'hotel', 'destination'],
            'health': ['health', 'medical', 'doctor', 'medicine', 'treatment', 'symptoms'],
            'technology': ['computer', 'technology', 'software', 'hardware', 'digital'],
            'sports': ['sports', 'game', 'team', 'player', 'match', 'exercise'],
            'education': ['learn', 'study', 'school', 'university', 'course', 'exam']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_keywords for keyword in keywords):
                return domain
        
        # 2. Use intent-based naming
        if question_intent['question_type'] != 'general':
            intent_categories = {
                'definition': 'concepts',
                'procedure': 'how_to',
                'comparison': 'comparisons',
                'recommendation': 'recommendations',
                'troubleshooting': 'troubleshooting',
                'factual': 'facts'
            }
            return intent_categories.get(question_intent['question_type'], 'general')
        
        # 3. Use most meaningful keyword
        if question_keywords:
            meaningful_words = sorted(list(question_keywords), key=lambda x: (-len(x), x))
            return meaningful_words[0]
        
        return "general"
    
    def classify_question(self, question: str) -> str:
        """
        Classify a full question into an appropriate category.
        
        Args:
            question: The full question text to classify
            
        Returns:
            The category name assigned to the question
        """
        if not question or not question.strip():
            return "empty"
        
        question = question.strip()
        
        # Display current categories for reference
        if len(self.categories) > 0:
            print(f"\nCurrent categories ({len(self.categories)}): {sorted(list(self.categories))}")
        
        # Get question embedding and analyze intent
        question_embedding = self._get_embedding(question)
        question_keywords = self._extract_keywords(question)
        question_intent = self._analyze_question_intent(question)
        
        print(f"Analyzing: '{question[:80]}...'")
        print(f"  Keywords: {list(question_keywords)[:5]}")
        print(f"  Intent: {question_intent['question_type']}")
        
        # Find best matching category
        best_category, similarity, match_reason = self._find_best_category_comprehensive(
            question, question_embedding
        )
        
        # Decision logic with higher threshold for better model
        if best_category and similarity >= self.similarity_threshold:
            category = best_category
            print(f"  -> Assigned to existing category '{category}' (score: {similarity:.3f}, reason: {match_reason})")
        else:
            # Create new category
            category = self._generate_category_name(question, question_embedding)
            
            # Ensure category name is unique
            original_category = category
            counter = 1
            while category in self.categories:
                category = f"{original_category}_{counter}"
                counter += 1
            
            self.categories.add(category)
            self.category_creation_history.append({
                'name': category,
                'first_question': question,
                'intent': question_intent['question_type'],
                'timestamp': len(self.category_creation_history) + 1
            })
            
            print(f"  -> NEW category '{category}' created (best existing similarity: {similarity:.3f})")
            if best_category and similarity > 0.5:
                print(f"     (closest existing category was '{best_category}')")
        
        # Add question to category and update keywords
        self.category_questions[category].append(question)
        self.category_keywords[category].update(question_keywords)
        
        # Update category embedding
        self._update_category_embedding(category)
        
        # Check for potential merges
        final_category = self._check_for_merges(category)
        
        return final_category
    
    def _check_for_merges(self, current_category: str) -> str:
        """Check if current category should be merged with existing categories."""
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
            
            # Combined similarity score
            combined_score = 0.7 * embedding_similarity + 0.3 * keyword_similarity
            
            if combined_score > best_merge_score and combined_score >= self.merge_threshold:
                best_merge_score = combined_score
                best_merge_candidate = other_category
        
        # Perform merge if candidate found
        if best_merge_candidate:
            print(f"  -> Auto-merge detected: '{current_category}' + '{best_merge_candidate}' (score: {best_merge_score:.3f})")
            
            # Merge into the category with more questions
            current_count = len(self.category_questions[current_category])
            candidate_count = len(self.category_questions[best_merge_candidate])
            
            if current_count >= candidate_count:
                return self._merge_categories(best_merge_candidate, current_category)
            else:
                return self._merge_categories(current_category, best_merge_candidate)
        
        return current_category
    
    def _merge_categories(self, source_category: str, target_category: str) -> str:
        """Merge source category into target category."""
        if source_category not in self.categories or target_category not in self.categories:
            return target_category
        
        # Move all questions from source to target
        self.category_questions[target_category].extend(self.category_questions[source_category])
        
        # Merge keywords
        self.category_keywords[target_category].update(self.category_keywords[source_category])
        
        # Remove source category
        self.categories.remove(source_category)
        del self.category_questions[source_category]
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
    
    def get_categories(self) -> Set[str]:
        """Get all current categories."""
        return self.categories.copy()
    
    def get_category_questions(self, category: str) -> List[str]:
        """Get all questions for a specific category."""
        return self.category_questions[category].copy()
    
    def display_category_summary(self):
        """Display a comprehensive summary of all current categories."""
        print(f"\n=== CATEGORY SUMMARY ({len(self.categories)} categories) ===")
        for i, category in enumerate(sorted(self.categories), 1):
            question_count = len(self.category_questions[category])
            keywords = list(self.category_keywords[category])[:5]
            
            print(f"{i:2d}. {category:<20} ({question_count:2d} questions)")
            print(f"    Keywords: {keywords}")
            
            # Show sample questions
            if self.category_questions[category]:
                sample_questions = self.category_questions[category][:2]
                for q in sample_questions:
                    print(f"    Q: {q[:60]}...")
            print()
        print("=" * 60)
    
    def save_model_state(self, filepath: str):
        """Save the classifier state to a file."""
        state = {
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'merge_threshold': self.merge_threshold,
            'categories': list(self.categories),
            'category_questions': dict(self.category_questions),
            'category_keywords': {k: list(v) for k, v in self.category_keywords.items()},
            'category_creation_history': self.category_creation_history,
            'merge_history': self.merge_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Classifier state saved to {filepath}")
    
    def load_model_state(self, filepath: str):
        """Load the classifier state from a file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.categories = set(state['categories'])
        self.category_questions = defaultdict(list, state['category_questions'])
        self.category_keywords = defaultdict(set, {k: set(v) for k, v in state['category_keywords'].items()})
        self.category_creation_history = state['category_creation_history']
        self.merge_history = state['merge_history']
        
        # Rebuild embeddings for existing categories
        print("Rebuilding category embeddings...")
        for category in self.categories:
            self._update_category_embedding(category)
        
        print(f"Classifier state loaded from {filepath}")