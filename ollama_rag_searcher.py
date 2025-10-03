from approach3_hybrid import HybridWebSearcher
import requests
import json
from typing import Dict, Any, List
import time

class OllamaRAGSearcher:
    def __init__(self, model_name: str = "llama3.2", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama RAG searcher
        
        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3.2', 'mistral', 'codellama')
            ollama_url: URL where Ollama is running
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.hybrid_searcher = HybridWebSearcher()
        
        # Test Ollama connection
        self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                print(f"✓ Connected to Ollama. Available models: {model_names}")
                
                if self.model_name not in model_names:
                    print(f"⚠ Warning: Model '{self.model_name}' not found. Available: {model_names}")
            else:
                print(f"✗ Ollama connection failed with status: {response.status_code}")
        except Exception as e:
            print(f"✗ Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def generate_with_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response using Ollama model
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            Generated text response
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Longer timeout for generation
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: Ollama responded with status {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be taking too long to respond."
        except Exception as e:
            return f"Error generating response: {e}"
    
    def create_rag_prompt(self, question: str, context: str, sources: List[str]) -> tuple:
        """
        Create a RAG prompt with context and sources
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context from web sources. 

Guidelines:
- Use ONLY the information provided in the context to answer questions
- If the context doesn't contain enough information, say so clearly
- Cite specific sources when making claims
- Be accurate and avoid speculation
- Provide comprehensive answers when possible
- If there are conflicting information in sources, mention it"""

        user_prompt = f"""Context from web sources:
{context}

Sources:
{chr(10).join([f"- {source}" for source in sources])}

Question: {question}

Please provide a comprehensive answer based on the context above. Include relevant details and cite sources where appropriate."""

        return system_prompt, user_prompt
    
    def search_and_answer(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Main function: Search web, retrieve relevant content, and generate answer using Ollama
        
        Args:
            question: The question to answer
            max_retries: Number of retries if web search fails
            
        Returns:
            Dictionary containing search results, context, and generated answer
        """
        print("🔍 Starting RAG-powered search and answer generation...")
        print("=" * 80)
        
        # Step 1: Perform hybrid web search
        search_result = None
        for attempt in range(max_retries + 1):
            try:
                search_result = self.hybrid_searcher.hybrid_search(question)
                if 'error' not in search_result:
                    break
                else:
                    print(f"Search attempt {attempt + 1} failed: {search_result['error']}")
                    if attempt < max_retries:
                        print("Retrying...")
                        time.sleep(2)
            except Exception as e:
                print(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    print("Retrying...")
                    time.sleep(2)
        
        if not search_result or 'error' in search_result:
            return {
                'question': question,
                'error': 'Failed to retrieve web content after multiple attempts',
                'answer': 'Sorry, I could not find relevant information to answer your question.'
            }
        
        # Step 2: Check if we have relevant content
        if not search_result.get('context') or len(search_result.get('relevant_content', [])) == 0:
            return {
                'question': question,
                'search_results': search_result,
                'error': 'No relevant content found',
                'answer': 'I could not find relevant information to answer your question.'
            }
        
        print(f"✓ Found {search_result['relevant_chunks_count']} relevant content chunks")
        print(f"✓ Context length: {len(search_result['context'])} characters")
        print(f"✓ Sources: {len(search_result['sources'])} unique URLs")
        
        # Step 3: Create RAG prompt
        system_prompt, user_prompt = self.create_rag_prompt(
            question, 
            search_result['context'], 
            search_result['sources']
        )
        
        # Step 4: Generate answer using Ollama
        print("🤖 Generating answer with Ollama...")
        answer = self.generate_with_ollama(user_prompt, system_prompt)
        
        # Step 5: Compile final result
        final_result = {
            'question': question,
            'search_results': search_result,
            'context': search_result['context'],
            'sources': search_result['sources'],
            'answer': answer,
            'model_used': self.model_name,
            'metadata': {
                'search_results_count': search_result.get('search_results_count', 0),
                'fetched_pages_count': search_result.get('fetched_pages_count', 0),
                'relevant_chunks_count': search_result.get('relevant_chunks_count', 0),
                'context_length': len(search_result.get('context', '')),
                'sources_count': len(search_result.get('sources', []))
            }
        }
        
        return final_result
    
    def display_rag_results(self, result: Dict[str, Any]):
        """
        Display the complete RAG results in a formatted way
        """
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("📊 SEARCH SUMMARY:")
        print("=" * 50)
        metadata = result.get('metadata', {})
        print(f"• Question: {result['question']}")
        print(f"• Model: {result.get('model_used', 'Unknown')}")
        print(f"• Search Results: {metadata.get('search_results_count', 0)}")
        print(f"• Successfully Fetched: {metadata.get('fetched_pages_count', 0)}")
        print(f"• Relevant Chunks: {metadata.get('relevant_chunks_count', 0)}")
        print(f"• Context Length: {metadata.get('context_length', 0)} characters")
        print(f"• Unique Sources: {metadata.get('sources_count', 0)}")
        
        print("\n🤖 AI ANSWER:")
        print("=" * 50)
        print(result['answer'])
        
        print(f"\n📚 SOURCES ({len(result.get('sources', []))}):")
        print("=" * 50)
        for i, source in enumerate(result.get('sources', []), 1):
            print(f"{i}. {source}")
        
        print(f"\n📄 RELEVANT CONTENT PREVIEW:")
        print("=" * 50)
        relevant_content = result.get('search_results', {}).get('relevant_content', [])
        for i, content in enumerate(relevant_content[:3], 1):  # Show first 3 chunks
            print(f"[{i}] {content['source']}")
            print(f"    {content['content'][:200]}...")
            print()
    
    def interactive_chat(self):
        """
        Start an interactive chat session with RAG-powered responses
        """
        print("🚀 Starting Interactive RAG Chat")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n💭 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', '']:
                    print("👋 Goodbye!")
                    break
                
                print("\n" + "="*80)
                result = self.search_and_answer(question)
                print("="*80)
                
                # Display just the answer in chat mode
                if 'error' not in result:
                    print(f"\n🤖 Answer:\n{result['answer']}")
                    print(f"\n📚 Sources: {len(result.get('sources', []))} URLs")
                else:
                    print(f"❌ {result.get('answer', 'An error occurred')}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

# Example usage and testing
if __name__ == "__main__":
    # Initialize the RAG searcher
    rag_searcher = OllamaRAGSearcher(model_name="llama3.2")  # Change model as needed
    
    # Test with a sample question
    test_question = "What are the latest developments in quantum computing?"
    
    print("🧪 Testing RAG Search with sample question...")
    result = rag_searcher.search_and_answer(test_question)
    rag_searcher.display_rag_results(result)
    
    # Uncomment to start interactive mode
    # rag_searcher.interactive_chat()