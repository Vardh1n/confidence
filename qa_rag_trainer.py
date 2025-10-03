import sqlite3
import json
import re
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from data_store import QADataStore
from ollama_rag_searcher import OllamaRAGSearcher
import time
import random

class QARAGTrainer:
    def __init__(self, 
                 source_db_path: str = "qa_database.db",
                 output_db_path: str = "qa_rag_results.db",
                 model_name: str = "llama3.2"):
        """
        Initialize the QA RAG trainer.
        
        Args:
            source_db_path: Path to the original QA database
            output_db_path: Path to store RAG results
            model_name: Ollama model to use
        """
        self.source_db = QADataStore(source_db_path)
        self.output_db_path = output_db_path
        self.rag_searcher = OllamaRAGSearcher(model_name=model_name)
        self.init_output_database()
    
    def init_output_database(self):
        """Initialize the output database with the required schema."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Main results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_results (
                    id INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    original_answer TEXT NOT NULL,
                    ai_generated_answer TEXT,
                    search_status TEXT DEFAULT 'pending',
                    total_articles INTEGER DEFAULT 0,
                    cited_articles INTEGER DEFAULT 0,
                    processing_time_seconds REAL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Articles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER,
                    website TEXT,
                    article_link TEXT,
                    content TEXT,
                    is_cited BOOLEAN DEFAULT 0,
                    relevance_score REAL,
                    FOREIGN KEY (result_id) REFERENCES rag_results (id)
                )
            ''')
            
            # Metadata table for tracking progress
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_questions INTEGER,
                    processed_questions INTEGER DEFAULT 0,
                    train_set_size INTEGER,
                    test_set_size INTEGER,
                    model_name TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'in_progress'
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_result_id ON rag_articles(result_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_status ON rag_results(search_status)')
            
            conn.commit()
    
    def load_qa_data(self) -> List[Dict]:
        """Load all QA data from the source database."""
        print("📚 Loading QA data from source database...")
        entries = self.source_db.get_all_entries()
        print(f"✓ Loaded {len(entries)} QA entries")
        return entries
    
    def perform_train_test_split(self, data: List[Dict], test_size: float = 0.1, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform train/test split on the QA data.
        
        Args:
            data: List of QA entries
            test_size: Fraction for test set (default 0.1 for 90:10 split)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        print(f"🔀 Performing {int((1-test_size)*100)}:{int(test_size*100)} train/test split...")
        
        if len(data) < 10:
            print("⚠ Warning: Dataset is small. Consider using all data for training.")
        
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        print(f"✓ Train set: {len(train_data)} questions")
        print(f"✓ Test set: {len(test_data)} questions")
        
        return train_data, test_data
    
    def extract_cited_sources(self, ai_answer: str, sources: List[str]) -> List[bool]:
        """
        Determine which sources were cited in the AI answer.
        
        Args:
            ai_answer: The AI-generated answer
            sources: List of source URLs
            
        Returns:
            List of boolean values indicating if each source was cited
        """
        cited = []
        ai_answer_lower = ai_answer.lower()
        
        for i, source in enumerate(sources):
            # Check various citation patterns
            is_cited = False
            
            # Check for source number references [1], [2], etc.
            if f"[{i+1}]" in ai_answer or f"({i+1})" in ai_answer:
                is_cited = True
            
            # Check for domain name mentions
            try:
                from urllib.parse import urlparse
                domain = urlparse(source).netloc.lower()
                if domain and domain in ai_answer_lower:
                    is_cited = True
            except:
                pass
            
            # Check for general source references
            if any(phrase in ai_answer_lower for phrase in [
                "according to", "source", "from", "as mentioned", 
                "cited", "reference", "based on"
            ]):
                # If there are source reference phrases and this is one of the first few sources
                if i < 3:  # Assume first 3 sources are more likely to be cited
                    is_cited = True
            
            cited.append(is_cited)
        
        return cited
    
    def process_single_question(self, qa_entry: Dict) -> Dict[str, Any]:
        """
        Process a single QA entry through the RAG pipeline.
        
        Args:
            qa_entry: Dictionary containing id, question, answer
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        question = qa_entry['question']
        original_answer = qa_entry['answer']
        entry_id = qa_entry['id']
        
        print(f"\n🔍 Processing Question ID {entry_id}:")
        print(f"Q: {question[:100]}...")
        
        try:
            # Use RAG searcher to get answer and sources
            rag_result = self.rag_searcher.search_and_answer(question)
            
            if 'error' in rag_result:
                return {
                    'id': entry_id,
                    'question': question,
                    'original_answer': original_answer,
                    'ai_generated_answer': None,
                    'articles': [],
                    'search_status': 'error',
                    'error_message': rag_result.get('error', 'Unknown error'),
                    'processing_time': time.time() - start_time
                }
            
            ai_answer = rag_result['answer']
            sources = rag_result.get('sources', [])
            relevant_content = rag_result.get('search_results', {}).get('relevant_content', [])
            
            # Determine which sources were cited
            cited_flags = self.extract_cited_sources(ai_answer, sources)
            
            # Prepare articles data
            articles = []
            for i, content_item in enumerate(relevant_content[:8]):  # Limit to 8 articles
                article = {
                    'website': self.extract_domain(content_item.get('source', '')),
                    'article_link': content_item.get('source', ''),
                    'content': content_item.get('content', ''),
                    'is_cited': cited_flags[i] if i < len(cited_flags) else False,
                    'relevance_score': content_item.get('relevance_score', 0.0)
                }
                articles.append(article)
            
            processing_time = time.time() - start_time
            
            print(f"✓ Generated answer ({len(ai_answer)} chars)")
            print(f"✓ Found {len(articles)} articles, {sum(cited_flags)} cited")
            print(f"✓ Processing time: {processing_time:.2f}s")
            
            return {
                'id': entry_id,
                'question': question,
                'original_answer': original_answer,
                'ai_generated_answer': ai_answer,
                'articles': articles,
                'search_status': 'completed',
                'error_message': None,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                'id': entry_id,
                'question': question,
                'original_answer': original_answer,
                'ai_generated_answer': None,
                'articles': [],
                'search_status': 'error',
                'error_message': str(e),
                'processing_time': time.time() - start_time
            }
    
    def extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return url
    
    def store_result(self, result: Dict[str, Any]):
        """Store a single result in the output database."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Insert main result
            cursor.execute('''
                INSERT OR REPLACE INTO rag_results 
                (id, question, original_answer, ai_generated_answer, search_status, 
                 total_articles, cited_articles, processing_time_seconds, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['id'],
                result['question'],
                result['original_answer'],
                result['ai_generated_answer'],
                result['search_status'],
                len(result['articles']),
                sum(1 for article in result['articles'] if article.get('is_cited', False)),
                result['processing_time'],
                result['error_message']
            ))
            
            # Delete existing articles for this result
            cursor.execute('DELETE FROM rag_articles WHERE result_id = ?', (result['id'],))
            
            # Insert articles
            for article in result['articles']:
                cursor.execute('''
                    INSERT INTO rag_articles 
                    (result_id, website, article_link, content, is_cited, relevance_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result['id'],
                    article['website'],
                    article['article_link'],
                    article['content'],
                    article['is_cited'],
                    article.get('relevance_score', 0.0)
                ))
            
            conn.commit()
    
    def update_metadata(self, total_questions: int, train_size: int, test_size: int, model_name: str):
        """Update training metadata."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_metadata 
                (total_questions, train_set_size, test_set_size, model_name)
                VALUES (?, ?, ?, ?)
            ''', (total_questions, train_size, test_size, model_name))
            conn.commit()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Get completed count
            cursor.execute("SELECT COUNT(*) FROM rag_results WHERE search_status != 'pending'")
            completed = cursor.fetchone()[0]
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM rag_results")
            total = cursor.fetchone()[0]
            
            # Get error count
            cursor.execute("SELECT COUNT(*) FROM rag_results WHERE search_status = 'error'")
            errors = cursor.fetchone()[0]
            
            # Get success count
            cursor.execute("SELECT COUNT(*) FROM rag_results WHERE search_status = 'completed'")
            success = cursor.fetchone()[0]
            
            return {
                'total': total,
                'completed': completed,
                'success': success,
                'errors': errors,
                'progress_percent': (completed / total * 100) if total > 0 else 0
            }
    
    def run_training_pipeline(self, test_size: float = 0.1, delay_between_requests: float = 2.0):
        """
        Run the complete training pipeline.
        
        Args:
            test_size: Fraction for test set
            delay_between_requests: Delay between API calls to avoid rate limiting
        """
        print("🚀 Starting QA RAG Training Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        qa_data = self.load_qa_data()
        if not qa_data:
            print("❌ No QA data found!")
            return
        
        # Step 2: Train/test split
        train_data, test_data = self.perform_train_test_split(qa_data, test_size)
        
        # Step 3: Update metadata
        self.update_metadata(
            len(qa_data), 
            len(train_data), 
            len(test_data), 
            self.rag_searcher.model_name
        )
        
        # Step 4: Process training data
        print(f"\n🎯 Processing {len(train_data)} training questions...")
        print("=" * 60)
        
        for i, qa_entry in enumerate(train_data, 1):
            print(f"\n[{i}/{len(train_data)}] Progress: {i/len(train_data)*100:.1f}%")
            
            # Process the question
            result = self.process_single_question(qa_entry)
            
            # Store the result
            self.store_result(result)
            
            # Show progress
            progress = self.get_progress()
            print(f"Overall progress: {progress['success']} success, {progress['errors']} errors")
            
            # Delay to avoid overwhelming the web search APIs
            if i < len(train_data):
                print(f"⏳ Waiting {delay_between_requests}s before next request...")
                time.sleep(delay_between_requests)
        
        # Step 5: Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final training summary."""
        print("\n" + "=" * 60)
        print("🎉 TRAINING PIPELINE COMPLETED")
        print("=" * 60)
        
        progress = self.get_progress()
        
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Get detailed statistics
            cursor.execute('''
                SELECT 
                    AVG(processing_time_seconds) as avg_time,
                    SUM(total_articles) as total_articles,
                    SUM(cited_articles) as total_cited,
                    AVG(cited_articles * 1.0 / total_articles) as avg_citation_rate
                FROM rag_results 
                WHERE search_status = 'completed'
            ''')
            stats = cursor.fetchone()
            
            print(f"📊 RESULTS SUMMARY:")
            print(f"  • Total Questions: {progress['total']}")
            print(f"  • Successfully Processed: {progress['success']}")
            print(f"  • Errors: {progress['errors']}")
            print(f"  • Success Rate: {progress['success']/progress['total']*100:.1f}%")
            
            if stats[0]:  # If we have timing data
                print(f"  • Average Processing Time: {stats[0]:.2f}s")
                print(f"  • Total Articles Collected: {stats[1] or 0}")
                print(f"  • Total Articles Cited: {stats[2] or 0}")
                print(f"  • Average Citation Rate: {(stats[3] or 0)*100:.1f}%")
            
            print(f"\n💾 Results saved to: {self.output_db_path}")
    
    def resume_training(self, delay_between_requests: float = 2.0):
        """Resume training from where it left off."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Find pending questions
            cursor.execute('''
                SELECT id, question, original_answer 
                FROM rag_results 
                WHERE search_status = 'pending'
                ORDER BY id
            ''')
            
            pending = cursor.fetchall()
            
        if not pending:
            print("✓ No pending questions found. Training appears complete.")
            return
        
        print(f"🔄 Resuming training with {len(pending)} pending questions...")
        
        for i, (entry_id, question, original_answer) in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] Resuming question ID {entry_id}")
            
            qa_entry = {
                'id': entry_id,
                'question': question,
                'answer': original_answer
            }
            
            result = self.process_single_question(qa_entry)
            self.store_result(result)
            
            if i < len(pending):
                time.sleep(delay_between_requests)
        
        self.print_final_summary()

# Example usage and testing
if __name__ == "__main__":
    # Initialize the trainer
    trainer = QARAGTrainer(
        source_db_path="qa_database.db",
        output_db_path="qa_rag_results.db",
        model_name="llama3.2"  # Change as needed
    )
    
    # Run the complete pipeline
    trainer.run_training_pipeline(test_size=0.1, delay_between_requests=3.0)
    
    # Or resume if interrupted
    # trainer.resume_training(delay_between_requests=3.0)
    
    print("QA RAG Trainer initialized. Uncomment the lines above to start training.")