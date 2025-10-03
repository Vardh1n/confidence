import re
from collections import Counter
from typing import List, Dict, Any
from duckduckgo_search import DDGS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

class SmartKeywordExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add common question words to stop words
        self.stop_words.update(['what', 'how', 'why', 'when', 'where', 'who', 'which', 'would', 'could', 'should'])
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract meaningful keywords from text using NLP techniques"""
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = word_tokenize(text)
        
        # Remove stop words and short words
        filtered_tokens = [word for word in tokens 
                          if word not in self.stop_words and len(word) > 2]
        
        # POS tagging to keep only meaningful parts of speech
        pos_tags = pos_tag(filtered_tokens)
        meaningful_words = [word for word, pos in pos_tags 
                           if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'VB', 'VBG', 'VBN']]
        
        # Count frequency and get most common
        word_freq = Counter(meaningful_words)
        
        # Extract bigrams for better context
        bigrams = [f"{meaningful_words[i]} {meaningful_words[i+1]}" 
                  for i in range(len(meaningful_words)-1)]
        bigram_freq = Counter(bigrams)
        
        # Combine single words and bigrams
        keywords = []
        
        # Add top bigrams first (they're usually more specific)
        for bigram, _ in bigram_freq.most_common(2):
            keywords.append(bigram)
        
        # Add top single words
        for word, _ in word_freq.most_common(max_keywords):
            if word not in ' '.join(keywords):  # Avoid duplicates
                keywords.append(word)
        
        return keywords[:max_keywords]

class DuckDuckGoSearcher:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search_multiple_pages(self, query: str, first_page_count: int = 6, second_page_count: int = 2) -> List[Dict[str, Any]]:
        """Search DuckDuckGo and return results from multiple pages"""
        
        all_results = []
        
        try:
            # Get more results than needed to simulate pagination
            total_needed = first_page_count + second_page_count
            results = list(self.ddgs.text(query, max_results=total_needed + 5))
            
            # Simulate first page (top 6 results)
            first_page = results[:first_page_count]
            for i, result in enumerate(first_page):
                result['page'] = 1
                result['position'] = i + 1
                all_results.append(result)
            
            # Simulate second page (next 2 results)
            second_page = results[first_page_count:first_page_count + second_page_count]
            for i, result in enumerate(second_page):
                result['page'] = 2
                result['position'] = i + 1
                all_results.append(result)
                
        except Exception as e:
            print(f"Search error: {e}")
            
        return all_results

def smart_web_search(text: str) -> Dict[str, Any]:
    """Main function to perform intelligent web search"""
    
    extractor = SmartKeywordExtractor()
    searcher = DuckDuckGoSearcher()
    
    # Extract keywords
    keywords = extractor.extract_keywords(text)
    search_query = ' '.join(keywords)
    
    print(f"Original text: {text}")
    print(f"Extracted keywords: {keywords}")
    print(f"Search query: {search_query}")
    print("-" * 50)
    
    # Perform search
    results = searcher.search_multiple_pages(search_query)
    
    return {
        'original_text': text,
        'keywords': keywords,
        'search_query': search_query,
        'results': results,
        'total_results': len(results)
    }

# Example usage
if __name__ == "__main__":
    test_text = "How does machine learning work in natural language processing applications?"
    
    search_results = smart_web_search(test_text)
    
    print(f"Found {search_results['total_results']} results:")
    for result in search_results['results']:
        print(f"Page {result['page']}, Position {result['position']}: {result['title']}")
        print(f"URL: {result['href']}")
        print(f"Snippet: {result['body'][:100]}...")
        print("-" * 30)