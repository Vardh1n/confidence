import sqlite3
import requests
from typing import List, Dict, Any, Optional
import time
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import json

class QADuckDuckGoCollector:
    def __init__(self, 
                 source_db_path: str = "qa_database.db",
                 output_db_path: str = "qa_ddg_articles.db"):
        """
        Initialize the QA DuckDuckGo article collector.
        
        Args:
            source_db_path: Path to the original QA database
            output_db_path: Path to store collected articles
        """
        self.source_db_path = source_db_path
        self.output_db_path = output_db_path
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.init_output_database()
    
    def init_output_database(self):
        """Initialize the output database with the required schema."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Main table with exactly 8 article columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_articles (
                    id INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    article_1_link TEXT,
                    article_1_title TEXT,
                    article_1_snippet TEXT,
                    article_2_link TEXT,
                    article_2_title TEXT,
                    article_2_snippet TEXT,
                    article_3_link TEXT,
                    article_3_title TEXT,
                    article_3_snippet TEXT,
                    article_4_link TEXT,
                    article_4_title TEXT,
                    article_4_snippet TEXT,
                    article_5_link TEXT,
                    article_5_title TEXT,
                    article_5_snippet TEXT,
                    article_6_link TEXT,
                    article_6_title TEXT,
                    article_6_snippet TEXT,
                    article_7_link TEXT,
                    article_7_title TEXT,
                    article_7_snippet TEXT,
                    article_8_link TEXT,
                    article_8_title TEXT,
                    article_8_snippet TEXT,
                    fetch_status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Metadata table for tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_questions INTEGER,
                    processed_questions INTEGER DEFAULT 0,
                    successful_fetches INTEGER DEFAULT 0,
                    failed_fetches INTEGER DEFAULT 0,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'in_progress'
                )
            ''')
            
            conn.commit()
            print(f"✓ Database initialized: {self.output_db_path}")
    
    def load_qa_data(self) -> List[Dict]:
        """Load all QA data from the source database."""
        print("📚 Loading QA data from source database...")
        
        with sqlite3.connect(self.source_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, question, answer FROM qa_entries ORDER BY id")
            rows = cursor.fetchall()
        
        qa_data = [
            {'id': row[0], 'question': row[1], 'answer': row[2]}
            for row in rows
        ]
        
        print(f"✓ Loaded {len(qa_data)} QA entries")
        return qa_data
    
    def search_duckduckgo(self, query: str, num_results: int = 20) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo and return results.
        
        Args:
            query: Search query
            num_results: Number of results to fetch (we'll get extras to ensure 8 good ones)
            
        Returns:
            List of search results with title, link, and snippet
        """
        try:
            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs:
                try:
                    # Extract link
                    link_elem = div.find('a', class_='result__a')
                    if not link_elem:
                        continue
                    
                    link = link_elem.get('href', '')
                    title = link_elem.get_text(strip=True)
                    
                    # Extract snippet
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if link and title:
                        results.append({
                            'title': title,
                            'link': link,
                            'snippet': snippet
                        })
                    
                    if len(results) >= num_results:
                        break
                        
                except Exception as e:
                    continue
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching DuckDuckGo: {e}")
            return []
    
    def get_exactly_8_results(self, query: str) -> List[Dict[str, str]]:
        """
        Get exactly 8 results: 6 from top, 2 from later pages.
        
        Args:
            query: Search query
            
        Returns:
            List of exactly 8 results or empty list if failed
        """
        print(f"🔍 Searching: {query[:80]}...")
        
        # Fetch more results to ensure we have enough
        all_results = self.search_duckduckgo(query, num_results=30)
        
        if len(all_results) < 8:
            print(f"⚠️ Only found {len(all_results)} results, need 8")
            return []
        
        # Select 6 from top (positions 0-5)
        top_6 = all_results[:6]
        
        # Select 2 from later results (positions 10+)
        # If we don't have enough later results, take what we can
        if len(all_results) >= 12:
            later_2 = all_results[10:12]
        elif len(all_results) >= 8:
            later_2 = all_results[6:8]
        else:
            later_2 = all_results[6:8]
        
        selected_8 = top_6 + later_2
        
        print(f"✓ Selected 8 results: 6 from top, 2 from positions {10 if len(all_results) >= 12 else 6}+")
        
        return selected_8
    
    def process_single_question(self, qa_entry: Dict) -> Dict[str, Any]:
        """
        Process a single QA entry and fetch 8 DuckDuckGo results.
        
        Args:
            qa_entry: Dictionary containing id, question, answer
            
        Returns:
            Dictionary with processing results
        """
        entry_id = qa_entry['id']
        question = qa_entry['question']
        answer = qa_entry['answer']
        
        print(f"\n{'='*80}")
        print(f"[ID {entry_id}] Processing question:")
        print(f"Q: {question[:100]}...")
        
        try:
            # Get exactly 8 results
            articles = self.get_exactly_8_results(question)
            
            if len(articles) != 8:
                return {
                    'id': entry_id,
                    'question': question,
                    'answer': answer,
                    'articles': [],
                    'fetch_status': 'failed',
                    'error_message': f'Could not fetch exactly 8 results, got {len(articles)}'
                }
            
            print(f"✓ Successfully fetched 8 articles")
            
            return {
                'id': entry_id,
                'question': question,
                'answer': answer,
                'articles': articles,
                'fetch_status': 'success',
                'error_message': None
            }
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                'id': entry_id,
                'question': question,
                'answer': answer,
                'articles': [],
                'fetch_status': 'error',
                'error_message': str(e)
            }
    
    def store_result(self, result: Dict[str, Any]):
        """Store a single result in the output database."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Prepare article data
            article_data = []
            for i in range(8):
                if i < len(result['articles']):
                    article = result['articles'][i]
                    article_data.extend([
                        article.get('link', ''),
                        article.get('title', ''),
                        article.get('snippet', '')
                    ])
                else:
                    article_data.extend(['', '', ''])  # Empty if not enough articles
            
            # Insert or replace the result
            cursor.execute('''
                INSERT OR REPLACE INTO qa_articles 
                (id, question, answer,
                 article_1_link, article_1_title, article_1_snippet,
                 article_2_link, article_2_title, article_2_snippet,
                 article_3_link, article_3_title, article_3_snippet,
                 article_4_link, article_4_title, article_4_snippet,
                 article_5_link, article_5_title, article_5_snippet,
                 article_6_link, article_6_title, article_6_snippet,
                 article_7_link, article_7_title, article_7_snippet,
                 article_8_link, article_8_title, article_8_snippet,
                 fetch_status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['id'],
                result['question'],
                result['answer'],
                *article_data,
                result['fetch_status'],
                result['error_message']
            ))
            
            conn.commit()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current collection progress."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM qa_articles")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM qa_articles WHERE fetch_status = 'success'")
            success = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM qa_articles WHERE fetch_status = 'failed'")
            failed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM qa_articles WHERE fetch_status = 'error'")
            errors = cursor.fetchone()[0]
            
            return {
                'total': total,
                'success': success,
                'failed': failed,
                'errors': errors,
                'progress_percent': (success / total * 100) if total > 0 else 0
            }
    
    def run_collection(self, delay_between_requests: float = 2.0):
        """
        Run the complete collection pipeline.
        
        Args:
            delay_between_requests: Delay between searches to avoid rate limiting
        """
        print("🚀 Starting QA DuckDuckGo Article Collection")
        print("=" * 80)
        
        # Load QA data
        qa_data = self.load_qa_data()
        if not qa_data:
            print("❌ No QA data found!")
            return
        
        # Update metadata
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO collection_metadata (total_questions)
                VALUES (?)
            ''', (len(qa_data),))
            conn.commit()
        
        # Process each question
        print(f"\n🎯 Processing {len(qa_data)} questions...")
        print("=" * 80)
        
        for i, qa_entry in enumerate(qa_data, 1):
            print(f"\n[{i}/{len(qa_data)}] Progress: {i/len(qa_data)*100:.1f}%")
            
            # Process the question
            result = self.process_single_question(qa_entry)
            
            # Store the result
            self.store_result(result)
            
            # Show progress
            progress = self.get_progress()
            print(f"Overall: {progress['success']} success, {progress['failed']} failed, {progress['errors']} errors")
            
            # Delay between requests
            if i < len(qa_data):
                print(f"⏳ Waiting {delay_between_requests}s before next search...")
                time.sleep(delay_between_requests)
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final collection summary."""
        print("\n" + "=" * 80)
        print("🎉 COLLECTION COMPLETED")
        print("=" * 80)
        
        progress = self.get_progress()
        
        print(f"📊 RESULTS SUMMARY:")
        print(f"  • Total Questions: {progress['total']}")
        print(f"  • Successfully Collected: {progress['success']}")
        print(f"  • Failed (< 8 results): {progress['failed']}")
        print(f"  • Errors: {progress['errors']}")
        print(f"  • Success Rate: {progress['success']/progress['total']*100:.1f}%")
        print(f"\n💾 Results saved to: {self.output_db_path}")
        print(f"📦 Total articles collected: {progress['success'] * 8}")
    
    def resume_collection(self, delay_between_requests: float = 2.0):
        """Resume collection from where it left off."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Find pending questions
            cursor.execute('''
                SELECT id, question, answer 
                FROM qa_articles 
                WHERE fetch_status = 'pending' OR fetch_status = 'failed'
                ORDER BY id
            ''')
            
            pending = cursor.fetchall()
        
        if not pending:
            print("✓ No pending questions found. Collection appears complete.")
            return
        
        print(f"🔄 Resuming collection with {len(pending)} pending questions...")
        
        for i, (entry_id, question, answer) in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] Resuming question ID {entry_id}")
            
            qa_entry = {
                'id': entry_id,
                'question': question,
                'answer': answer
            }
            
            result = self.process_single_question(qa_entry)
            self.store_result(result)
            
            if i < len(pending):
                time.sleep(delay_between_requests)
        
        self.print_final_summary()

# Example usage
if __name__ == "__main__":
    # Initialize the collector
    collector = QADuckDuckGoCollector(
        source_db_path="qa_database.db",
        output_db_path="qa_ddg_articles.db"
    )
    
    # Run the collection
    collector.run_collection(delay_between_requests=8)
    
    # Or resume if interrupted
    # collector.resume_collection(delay_between_requests=3.0)