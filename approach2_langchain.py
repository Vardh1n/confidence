from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import time

class LangChainWebSearcher:
    def __init__(self):
        # Initialize DuckDuckGo search
        self.search_wrapper = DuckDuckGoSearchAPIWrapper(
            region="us-en",
            time="y",  # Past year
            max_results=8
        )
        self.search_tool = DuckDuckGoSearchRun(api_wrapper=self.search_wrapper)
        
        # Initialize embeddings for semantic search
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def search_and_fetch_content(self, query: str) -> List[Dict[str, Any]]:
        """Search web and fetch full content from results"""
        
        print(f"Searching for: {query}")
        
        # Perform search
        search_results = self.search_tool.run(query)
        
        # Parse search results (DuckDuckGo returns formatted string)
        results = self._parse_search_results(search_results)
        
        # Fetch full content from each URL
        enriched_results = []
        for result in results:
            try:
                content = self._fetch_webpage_content(result['url'])
                if content:
                    result['full_content'] = content
                    result['content_length'] = len(content)
                    enriched_results.append(result)
                time.sleep(1)  # Be respectful to servers
            except Exception as e:
                print(f"Error fetching {result['url']}: {e}")
                continue
        
        return enriched_results
    
    def _parse_search_results(self, search_string: str) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo search results from string format"""
        results = []
        
        # Simple parsing - in practice, you might want more robust parsing
        lines = search_string.split('\n')
        current_result = {}
        
        for line in lines:
            if line.startswith('Title: '):
                if current_result:
                    results.append(current_result)
                current_result = {'title': line[7:]}
            elif line.startswith('Link: '):
                current_result['url'] = line[6:]
            elif line.startswith('Snippet: '):
                current_result['snippet'] = line[9:]
        
        if current_result:
            results.append(current_result)
        
        return results
    
    def _fetch_webpage_content(self, url: str) -> str:
        """Fetch and extract text content from webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Linux; x86_64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit content length
            
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return ""
    
    def create_knowledge_base(self, search_results: List[Dict[str, Any]]) -> FAISS:
        """Create a vector store from search results for semantic search"""
        
        documents = []
        for result in search_results:
            if 'full_content' in result:
                # Split content into chunks
                chunks = self.text_splitter.split_text(result['full_content'])
                
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'title': result['title'],
                            'url': result['url'],
                            'snippet': result.get('snippet', '')
                        }
                    )
                    documents.append(doc)
        
        if not documents:
            return None
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore
    
    def answer_question_with_search(self, question: str) -> Dict[str, Any]:
        """Search web and answer question using retrieved content"""
        
        # Search and fetch content
        search_results = self.search_and_fetch_content(question)
        
        if not search_results:
            return {
                'question': question,
                'answer': "No relevant content found.",
                'sources': []
            }
        
        # Create knowledge base
        vectorstore = self.create_knowledge_base(search_results)
        
        if not vectorstore:
            return {
                'question': question,
                'answer': "Could not process search results.",
                'sources': [r['url'] for r in search_results]
            }
        
        # Create a simple prompt template
        prompt_template = """
        Use the following pieces of context to answer the question. 
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Get relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Simple answer generation (without LLM)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        return {
            'question': question,
            'context': context,
            'search_results': search_results,
            'relevant_documents': len(relevant_docs),
            'sources': list(set([doc.metadata['url'] for doc in relevant_docs]))
        }

# Example usage
if __name__ == "__main__":
    searcher = LangChainWebSearcher()
    
    question = "What are the latest developments in quantum computing?"
    
    result = searcher.answer_question_with_search(question)
    
    print(f"Question: {result['question']}")
    print(f"Found {len(result['search_results'])} search results")
    print(f"Used {result['relevant_documents']} relevant documents")
    print(f"Sources: {result['sources']}")
    print(f"Context preview: {result['context'][:500]}...")