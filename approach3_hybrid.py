from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
import time
import re

class HybridWebSearcher:
    def __init__(self):
        # Initialize embeddings for semantic search
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.ddgs = DDGS()
    
    def search_duckduckgo(self, question: str, first_page_count: int = 6, second_page_count: int = 2) -> List[Dict[str, Any]]:
        """
        Search DuckDuckGo with full question, get top 6 from page 1 and top 2 from page 2
        Returns only the links and basic metadata
        """
        print(f"Searching DuckDuckGo for: {question}")
        
        try:
            # Get more results to simulate pagination
            total_needed = first_page_count + second_page_count
            all_results = list(self.ddgs.text(question, max_results=total_needed + 5))
            
            selected_results = []
            
            # First page results (top 6)
            for i, result in enumerate(all_results[:first_page_count]):
                selected_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'page': 1,
                    'position': i + 1
                })
            
            # Second page results (next 2)
            second_page_start = first_page_count
            second_page_end = second_page_start + second_page_count
            
            for i, result in enumerate(all_results[second_page_start:second_page_end]):
                selected_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'page': 2,
                    'position': i + 1
                })
            
            print(f"Retrieved {len(selected_results)} search results")
            return selected_results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def fetch_webpage_content(self, url: str) -> str:
        """
        Fetch and extract clean text content from a webpage
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Linux; x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button']):
                element.decompose()
            
            # Try to find main content area
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=re.compile(r'content|article|post|main', re.I)) or
                soup.find('body')
            )
            
            if main_content:
                text = main_content.get_text()
            else:
                text = soup.get_text()
            
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove extra whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # Limit content length but be more generous
            return clean_text[:8000]
            
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return ""
    
    def fetch_all_webpages(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch webpage content for all search results
        """
        enriched_results = []
        
        for i, result in enumerate(search_results):
            print(f"Fetching content from {i+1}/{len(search_results)}: {result['url']}")
            
            content = self.fetch_webpage_content(result['url'])
            
            if content:
                result['full_content'] = content
                result['content_length'] = len(content)
                enriched_results.append(result)
                print(f"  ✓ Fetched {len(content)} characters")
            else:
                print(f"  ✗ Failed to fetch content")
            
            # Be respectful to servers
            time.sleep(1)
        
        print(f"Successfully fetched content from {len(enriched_results)}/{len(search_results)} pages")
        return enriched_results
    
    def create_vector_store(self, enriched_results: List[Dict[str, Any]]) -> FAISS:
        """
        Create vector embeddings from fetched content
        """
        print("Creating vector embeddings...")
        
        documents = []
        
        for result in enriched_results:
            if 'full_content' in result and result['full_content']:
                # Split content into chunks
                chunks = self.text_splitter.split_text(result['full_content'])
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'title': result['title'],
                            'url': result['url'],
                            'snippet': result['snippet'],
                            'page': result['page'],
                            'position': result['position'],
                            'chunk_id': i,
                            'source': f"{result['title']} (Page {result['page']}, Position {result['position']})"
                        }
                    )
                    documents.append(doc)
        
        if not documents:
            print("No documents to create vector store")
            return None
        
        print(f"Created {len(documents)} document chunks")
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore
    
    def get_relevant_content(self, vectorstore: FAISS, question: str, k: int = 5) -> List[Document]:
        """
        Get most relevant content chunks for the question using semantic similarity
        """
        if not vectorstore:
            return []
        
        print(f"Finding {k} most relevant content chunks...")
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        relevant_docs = retriever.get_relevant_documents(question)
        return relevant_docs
    
    def hybrid_search(self, question: str) -> Dict[str, Any]:
        """
        Main hybrid search function that combines DuckDuckGo search with semantic retrieval
        """
        print("=" * 60)
        print("HYBRID WEB SEARCH")
        print("=" * 60)
        
        # Step 1: Search DuckDuckGo with full question
        search_results = self.search_duckduckgo(question)
        
        if not search_results:
            return {
                'question': question,
                'error': 'No search results found',
                'results': []
            }
        
        # Step 2: Fetch webpage content for all results
        enriched_results = self.fetch_all_webpages(search_results)
        
        if not enriched_results:
            return {
                'question': question,
                'error': 'Could not fetch any webpage content',
                'search_results': search_results
            }
        
        # Step 3: Create vector embeddings
        vectorstore = self.create_vector_store(enriched_results)
        
        if not vectorstore:
            return {
                'question': question,
                'error': 'Could not create vector embeddings',
                'enriched_results': enriched_results
            }
        
        # Step 4: Get relevant content based on original question
        relevant_docs = self.get_relevant_content(vectorstore, question)
        
        # Compile results
        result = {
            'question': question,
            'search_results_count': len(search_results),
            'fetched_pages_count': len(enriched_results),
            'relevant_chunks_count': len(relevant_docs),
            'search_results': search_results,
            'enriched_results': enriched_results,
            'relevant_content': [
                {
                    'content': doc.page_content,
                    'source': doc.metadata['source'],
                    'url': doc.metadata['url'],
                    'title': doc.metadata['title']
                }
                for doc in relevant_docs
            ],
            'sources': list(set([doc.metadata['url'] for doc in relevant_docs])),
            'context': '\n\n'.join([doc.page_content for doc in relevant_docs])
        }
        
        return result
    
    def display_results(self, result: Dict[str, Any]):
        """
        Display search results in a formatted way
        """
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\nQuestion: {result['question']}")
        print(f"Search Results: {result['search_results_count']}")
        print(f"Successfully Fetched: {result['fetched_pages_count']}")
        print(f"Relevant Chunks: {result['relevant_chunks_count']}")
        print(f"Unique Sources: {len(result['sources'])}")
        
        print("\n" + "="*50)
        print("MOST RELEVANT CONTENT:")
        print("="*50)
        
        for i, content in enumerate(result['relevant_content']):
            print(f"\n[{i+1}] {content['source']}")
            print(f"URL: {content['url']}")
            print(f"Content: {content['content'][:300]}...")
            print("-" * 30)

# Example usage
if __name__ == "__main__":
    searcher = HybridWebSearcher()
    
    # Test with a complex question
    question = "What are the latest developments in quantum computing and how do they compare to classical computing?"
    
    result = searcher.hybrid_search(question)
    searcher.display_results(result)
    
    # You can also access the structured data
    print(f"\nFull context length: {len(result.get('context', ''))}")
    print(f"Sources: {result.get('sources', [])}")