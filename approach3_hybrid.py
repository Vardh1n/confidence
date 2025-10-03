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
import asyncio
import aiohttp
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
        # Browser-like headers to appear more legitimate
        self.headers_list = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
            },
            {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
            }
        ]
    
    def search_duckduckgo(self, question: str, target_count: int = 8) -> List[Dict[str, Any]]:
        """
        Search DuckDuckGo and get enough results to ensure we can fetch target_count articles
        """
        print(f"Searching DuckDuckGo for: {question}")
        
        try:
            # Get more results than needed to account for failed fetches
            max_results = target_count * 3  # Get 3x more to account for failures
            all_results = list(self.ddgs.text(question, max_results=max_results))
            
            search_results = []
            
            for i, result in enumerate(all_results):
                search_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'position': i + 1
                })
            
            print(f"Retrieved {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def fetch_webpage_content_with_retry(self, url: str, max_retries: int = 2) -> str:
        """
        Fetch webpage content with retry logic and browser-like behavior
        """
        for attempt in range(max_retries + 1):
            try:
                # Use random headers to appear more like different browsers
                headers = random.choice(self.headers_list).copy()
                
                # Add some randomization to make requests look more natural
                time.sleep(random.uniform(0.5, 1.5))
                
                session = requests.Session()
                session.headers.update(headers)
                
                response = session.get(
                    url, 
                    timeout=20,
                    allow_redirects=True,
                    verify=True
                )
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'noscript']):
                    element.decompose()
                
                # Try to find main content area with more comprehensive selectors
                main_content = (
                    soup.find('main') or 
                    soup.find('article') or 
                    soup.find('div', class_=re.compile(r'content|article|post|main|body|text', re.I)) or
                    soup.find('div', id=re.compile(r'content|article|post|main|body', re.I)) or
                    soup.find('section', class_=re.compile(r'content|article|post|main', re.I)) or
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
                
                # Ensure we have substantial content
                if len(clean_text) < 100:
                    raise Exception("Content too short, might be blocked or empty")
                
                # Limit content length but be more generous
                return clean_text[:10000]
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries:
                    print(f"  Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"  Giving up on {url} after {max_retries + 1} attempts")
                    return ""
        
        return ""
    
    def fetch_single_webpage(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch a single webpage with retry logic
        """
        url = result['url']
        print(f"Fetching: {result['title'][:50]}...")
        
        content = self.fetch_webpage_content_with_retry(url)
        
        if content:
            result['full_content'] = content
            result['content_length'] = len(content)
            result['fetch_success'] = True
            print(f"  ✓ Success: {len(content)} characters")
        else:
            result['fetch_success'] = False
            print(f"  ✗ Failed after retries")
        
        return result
    
    def fetch_all_webpages_parallel(self, search_results: List[Dict[str, Any]], target_count: int = 8) -> List[Dict[str, Any]]:
        """
        Fetch webpage content for search results in parallel until we get target_count successful fetches
        """
        print(f"Fetching webpages in parallel (target: {target_count} successful fetches)...")
        
        enriched_results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all fetch tasks
            future_to_result = {
                executor.submit(self.fetch_single_webpage, result): result 
                for result in search_results
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_result):
                result = future.result()
                
                if result['fetch_success']:
                    enriched_results.append(result)
                    print(f"Progress: {len(enriched_results)}/{target_count} successful fetches")
                    
                    # Stop once we have enough successful fetches
                    if len(enriched_results) >= target_count:
                        print("Target count reached, cancelling remaining tasks...")
                        # Cancel remaining futures
                        for remaining_future in future_to_result:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
        
        print(f"Successfully fetched content from {len(enriched_results)} pages")
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
                            'position': result['position'],
                            'chunk_id': i,
                            'source': f"{result['title']} (Position {result['position']})"
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
    
    def get_relevant_content(self, vectorstore: FAISS, question: str, k: int = 8) -> List[Document]:
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
    
    def hybrid_search(self, question: str, target_articles: int = 8) -> Dict[str, Any]:
        """
        Main hybrid search function that combines DuckDuckGo search with semantic retrieval
        """
        print("=" * 60)
        print("HYBRID WEB SEARCH WITH PARALLEL FETCHING")
        print("=" * 60)
        
        # Step 1: Search DuckDuckGo with full question
        search_results = self.search_duckduckgo(question, target_articles)
        
        if not search_results:
            return {
                'question': question,
                'error': 'No search results found',
                'results': []
            }
        
        # Step 2: Fetch webpage content in parallel until we get target_articles
        enriched_results = self.fetch_all_webpages_parallel(search_results, target_articles)
        
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
        relevant_docs = self.get_relevant_content(vectorstore, question, k=min(10, len(enriched_results) * 2))
        
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
    
    result = searcher.hybrid_search(question, target_articles=8)
    searcher.display_results(result)
    
    # You can also access the structured data
    print(f"\nFull context length: {len(result.get('context', ''))}")
    print(f"Sources: {result.get('sources', [])}")