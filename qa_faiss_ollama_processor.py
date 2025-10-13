import sqlite3
import numpy as np
from typing import Dict, Any, List, Tuple
import time
from sentence_transformers import SentenceTransformer
import faiss
import requests

class QAFAISSOllamaProcessor:
    def __init__(self,
                 input_db_path: str = "qa_ddg_articles.db",
                 output_db_path: str = "qa_faiss_ollama_results.db",
                 model_name: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the FAISS + Ollama processor for QA data.
        
        Args:
            input_db_path: Path to the QA articles database
            output_db_path: Path to store results with chunk counts
            model_name: Ollama model name
            ollama_url: Ollama server URL
            embedding_model: Sentence transformer model for embeddings
        """
        self.input_db_path = input_db_path
        self.output_db_path = output_db_path
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Initialize embedding model
        print(f"📦 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✓ Embedding model loaded")
        
        # Test Ollama connection
        self._test_ollama_connection()
        
        # Initialize output database
        self.init_output_database()
    
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
    
    def init_output_database(self):
        """Initialize output database with chunk count columns and generated answer."""
        with sqlite3.connect(self.output_db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            cursor = conn.cursor()
            
            # Create main results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_faiss_results (
                    id INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    article_1_link TEXT,
                    article_1_title TEXT,
                    article_1_snippet TEXT,
                    article_1_no_of_chunks INTEGER DEFAULT 0,
                    article_2_link TEXT,
                    article_2_title TEXT,
                    article_2_snippet TEXT,
                    article_2_no_of_chunks INTEGER DEFAULT 0,
                    article_3_link TEXT,
                    article_3_title TEXT,
                    article_3_snippet TEXT,
                    article_3_no_of_chunks INTEGER DEFAULT 0,
                    article_4_link TEXT,
                    article_4_title TEXT,
                    article_4_snippet TEXT,
                    article_4_no_of_chunks INTEGER DEFAULT 0,
                    article_5_link TEXT,
                    article_5_title TEXT,
                    article_5_snippet TEXT,
                    article_5_no_of_chunks INTEGER DEFAULT 0,
                    article_6_link TEXT,
                    article_6_title TEXT,
                    article_6_snippet TEXT,
                    article_6_no_of_chunks INTEGER DEFAULT 0,
                    article_7_link TEXT,
                    article_7_title TEXT,
                    article_7_snippet TEXT,
                    article_7_no_of_chunks INTEGER DEFAULT 0,
                    article_8_link TEXT,
                    article_8_title TEXT,
                    article_8_snippet TEXT,
                    article_8_no_of_chunks INTEGER DEFAULT 0,
                    ollama_generated_answer TEXT,
                    total_chunks_used INTEGER DEFAULT 0,
                    processing_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_questions INTEGER,
                    processed_questions INTEGER DEFAULT 0,
                    successful_processing INTEGER DEFAULT 0,
                    failed_processing INTEGER DEFAULT 0,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'in_progress'
                )
            ''')
            
            conn.commit()
            print(f"✓ Output database initialized: {self.output_db_path}")
    
    def load_qa_entry(self, entry_id: int) -> Dict[str, Any]:
        """Load a single QA entry with all articles from input database."""
        with sqlite3.connect(self.input_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, question, answer,
                       article_1_link, article_1_title, article_1_snippet,
                       article_2_link, article_2_title, article_2_snippet,
                       article_3_link, article_3_title, article_3_snippet,
                       article_4_link, article_4_title, article_4_snippet,
                       article_5_link, article_5_title, article_5_snippet,
                       article_6_link, article_6_title, article_6_snippet,
                       article_7_link, article_7_title, article_7_snippet,
                       article_8_link, article_8_title, article_8_snippet
                FROM qa_articles
                WHERE id = ? AND fetch_status = 'success'
            ''', (entry_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse into structured format
            entry = {
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'articles': []
            }
            
            # Extract 8 articles
            for i in range(8):
                base_idx = 3 + (i * 3)
                article = {
                    'index': i + 1,
                    'link': row[base_idx],
                    'title': row[base_idx + 1],
                    'snippet': row[base_idx + 2]
                }
                entry['articles'].append(article)
            
            return entry
    
    def get_all_entry_ids(self) -> List[int]:
        """Get all successfully fetched entry IDs from input database."""
        with sqlite3.connect(self.input_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM qa_articles 
                WHERE fetch_status = 'success'
                ORDER BY id
            ''')
            return [row[0] for row in cursor.fetchall()]
    
    def create_chunks_from_snippets(self, articles: List[Dict]) -> Tuple[List[str], List[int]]:
        """
        Create text chunks from article snippets.
        Each snippet is treated as one chunk to track article usage.
        
        Returns:
            Tuple of (chunks, article_indices)
        """
        chunks = []
        article_indices = []
        
        for article in articles:
            snippet = article.get('snippet', '').strip()
            if snippet:
                # Create chunk with metadata
                chunk_text = f"[Article {article['index']}: {article['title']}]\n{snippet}"
                chunks.append(chunk_text)
                article_indices.append(article['index'])
        
        return chunks, article_indices
    
    def build_faiss_index(self, chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build FAISS index from text chunks.
        
        Returns:
            Tuple of (faiss_index, embeddings)
        """
        if not chunks:
            return None, None
        
        # Generate embeddings
        print(f"  🔢 Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings)
        
        print(f"  ✓ FAISS index built with {index.ntotal} vectors")
        
        return index, embeddings
    
    def retrieve_relevant_chunks(self, 
                                 question: str, 
                                 faiss_index: faiss.Index,
                                 chunks: List[str],
                                 article_indices: List[int],
                                 top_k: int = 5) -> Tuple[List[str], List[int]]:
        """
        Retrieve top-k relevant chunks using FAISS.
        
        Returns:
            Tuple of (selected_chunks, selected_article_indices)
        """
        # Encode question
        question_embedding = self.embedding_model.encode([question])[0]
        question_embedding = np.array([question_embedding]).astype('float32')
        faiss.normalize_L2(question_embedding)
        
        # Search FAISS index
        distances, indices = faiss_index.search(question_embedding, min(top_k, len(chunks)))
        
        # Get selected chunks and their article indices
        selected_chunks = [chunks[idx] for idx in indices[0]]
        selected_article_indices = [article_indices[idx] for idx in indices[0]]
        
        return selected_chunks, selected_article_indices
    
    def count_chunks_per_article(self, selected_article_indices: List[int]) -> Dict[int, int]:
        """
        Count how many chunks were selected from each article.
        
        Returns:
            Dictionary mapping article_index (1-8) to chunk count
        """
        chunk_counts = {i: 0 for i in range(1, 9)}
        
        for article_idx in selected_article_indices:
            chunk_counts[article_idx] += 1
        
        return chunk_counts
    
    def generate_with_ollama(self, 
                            question: str, 
                            context_chunks: List[str],
                            article_indices: List[int]) -> str:
        """
        Generate answer using Ollama with retrieved context.
        
        Args:
            question: The question to answer
            context_chunks: Retrieved relevant chunks
            article_indices: Article indices for each chunk
            
        Returns:
            Generated answer text
        """
        # Build context from chunks
        context_text = "\n\n".join([
            f"[Source {article_indices[i]}] {chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context.

Guidelines:
- Use ONLY the information provided in the context
- Be accurate and concise
- If context doesn't contain enough information, state that clearly
- Cite sources when making specific claims"""

        user_prompt = f"""Context from web sources:

{context_text}

Question: {question}

Please provide a comprehensive answer based on the context above."""

        try:
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: Ollama responded with status {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out"
        except Exception as e:
            return f"Error generating response: {e}"
    
    def process_single_question(self, entry_id: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a single QA entry: load articles, apply FAISS, generate answer.
        
        Args:
            entry_id: ID of the QA entry
            top_k: Number of top chunks to retrieve
            
        Returns:
            Processing result dictionary
        """
        print(f"\n{'='*80}")
        print(f"[ID {entry_id}] Processing question...")
        
        try:
            # Load QA entry
            entry = self.load_qa_entry(entry_id)
            if not entry:
                return {
                    'id': entry_id,
                    'status': 'error',
                    'error': 'Entry not found or not successfully fetched'
                }
            
            print(f"Q: {entry['question'][:100]}...")
            
            # Create chunks from snippets
            chunks, article_indices = self.create_chunks_from_snippets(entry['articles'])
            print(f"  📝 Created {len(chunks)} chunks from articles")
            
            if not chunks:
                return {
                    'id': entry_id,
                    'status': 'error',
                    'error': 'No valid chunks created from articles'
                }
            
            # Build FAISS index
            faiss_index, embeddings = self.build_faiss_index(chunks)
            
            # Retrieve relevant chunks
            print(f"  🔍 Retrieving top-{top_k} relevant chunks...")
            selected_chunks, selected_article_indices = self.retrieve_relevant_chunks(
                entry['question'],
                faiss_index,
                chunks,
                article_indices,
                top_k=top_k
            )
            
            # Count chunks per article
            chunk_counts = self.count_chunks_per_article(selected_article_indices)
            print(f"  📊 Chunk distribution: {[chunk_counts[i] for i in range(1, 9)]}")
            
            # Generate answer with Ollama
            print(f"  🤖 Generating answer with Ollama...")
            generated_answer = self.generate_with_ollama(
                entry['question'],
                selected_chunks,
                selected_article_indices
            )
            
            print(f"  ✓ Answer generated ({len(generated_answer)} chars)")
            
            return {
                'id': entry_id,
                'status': 'success',
                'entry': entry,
                'chunk_counts': chunk_counts,
                'generated_answer': generated_answer,
                'total_chunks_used': sum(1 for c in chunk_counts.values() if c > 0)
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                'id': entry_id,
                'status': 'error',
                'error': str(e)
            }
    
    def store_result(self, result: Dict[str, Any]):
        """Store processing result in output database."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            if result['status'] == 'error':
                cursor.execute('''
                    INSERT OR REPLACE INTO qa_faiss_results 
                    (id, question, answer, processing_status, error_message)
                    VALUES (?, ?, ?, 'error', ?)
                ''', (result['id'], '', '', result['error']))
            else:
                entry = result['entry']
                chunk_counts = result['chunk_counts']
                
                # Prepare article data
                article_data = []
                for i in range(1, 9):
                    article = entry['articles'][i-1]
                    article_data.extend([
                        article['link'],
                        article['title'],
                        article['snippet'],
                        chunk_counts[i]
                    ])
                
                cursor.execute('''
                    INSERT OR REPLACE INTO qa_faiss_results 
                    (id, question, answer,
                     article_1_link, article_1_title, article_1_snippet, article_1_no_of_chunks,
                     article_2_link, article_2_title, article_2_snippet, article_2_no_of_chunks,
                     article_3_link, article_3_title, article_3_snippet, article_3_no_of_chunks,
                     article_4_link, article_4_title, article_4_snippet, article_4_no_of_chunks,
                     article_5_link, article_5_title, article_5_snippet, article_5_no_of_chunks,
                     article_6_link, article_6_title, article_6_snippet, article_6_no_of_chunks,
                     article_7_link, article_7_title, article_7_snippet, article_7_no_of_chunks,
                     article_8_link, article_8_title, article_8_snippet, article_8_no_of_chunks,
                     ollama_generated_answer, total_chunks_used, processing_status)
                    VALUES (?, ?, ?, 
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'success')
                ''', (
                    result['id'],
                    entry['question'],
                    entry['answer'],
                    *article_data,
                    result['generated_answer'],
                    result['total_chunks_used']
                ))
            
            conn.commit()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM qa_faiss_results")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM qa_faiss_results WHERE processing_status = 'success'")
            success = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM qa_faiss_results WHERE processing_status = 'error'")
            errors = cursor.fetchone()[0]
            
            return {
                'total': total,
                'success': success,
                'errors': errors,
                'progress_percent': (success / total * 100) if total > 0 else 0
            }
    
    def run_processing(self, top_k: int = 5, delay_between_questions: float = 1.0):
        """
        Run complete processing pipeline for all QA entries.
        
        Args:
            top_k: Number of top chunks to retrieve per question
            delay_between_questions: Delay between processing questions
        """
        print("🚀 Starting FAISS + Ollama QA Processing")
        print("=" * 80)
        
        # Get all entry IDs
        entry_ids = self.get_all_entry_ids()
        print(f"📚 Found {len(entry_ids)} successfully fetched QA entries")
        
        if not entry_ids:
            print("❌ No entries to process!")
            return
        
        # Update metadata
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO processing_metadata (total_questions)
                VALUES (?)
            ''', (len(entry_ids),))
            conn.commit()
        
        # Process each entry
        print(f"\n🎯 Processing {len(entry_ids)} questions with top_k={top_k}...")
        print("=" * 80)
        
        for i, entry_id in enumerate(entry_ids, 1):
            print(f"\n[{i}/{len(entry_ids)}] Progress: {i/len(entry_ids)*100:.1f}%")
            
            # Process the question
            result = self.process_single_question(entry_id, top_k=top_k)
            
            # Store the result
            self.store_result(result)
            
            # Show progress
            progress = self.get_progress()
            print(f"Overall: {progress['success']} success, {progress['errors']} errors")
            
            # Delay between requests
            if i < len(entry_ids):
                time.sleep(delay_between_questions)
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final processing summary."""
        print("\n" + "=" * 80)
        print("🎉 PROCESSING COMPLETED")
        print("=" * 80)
        
        progress = self.get_progress()
        
        print(f"📊 RESULTS SUMMARY:")
        print(f"  • Total Questions: {progress['total']}")
        print(f"  • Successfully Processed: {progress['success']}")
        print(f"  • Errors: {progress['errors']}")
        print(f"  • Success Rate: {progress['success']/progress['total']*100:.1f}%")
        
        # Get chunk usage statistics
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(total_chunks_used) FROM qa_faiss_results 
                WHERE processing_status = 'success'
            ''')
            avg_chunks = cursor.fetchone()[0] or 0
            
            print(f"  • Average Chunks Used: {avg_chunks:.2f}")
        
        print(f"\n💾 Results saved to: {self.output_db_path}")

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = QAFAISSOllamaProcessor(
        input_db_path="qa_ddg_articles.db",
        output_db_path="qa_faiss_ollama_results.db",
        model_name="gemma3n:latest",
        embedding_model="all-MiniLM-L6-v2"
    )

    # Run processing with top-10 chunks retrieval
    processor.run_processing(top_k=10, delay_between_questions=1.0)