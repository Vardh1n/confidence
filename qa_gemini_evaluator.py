import sqlite3
import json
import time
from typing import Dict, Any, List, Tuple
import google.generativeai as genai
from pathlib import Path
import shutil

class QAGeminiEvaluator:
    def __init__(self,
                 input_db_path: str = "qa_faiss_ollama_results.db",
                 output_db_path: str = "qa_faiss_ollama_evaluated.db",
                 gemini_api_key: str = None,
                 model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini evaluator for QA results.
        
        Args:
            input_db_path: Path to the FAISS+Ollama results database
            output_db_path: Path to store evaluated results
            gemini_api_key: Google Gemini API key
            model_name: Gemini model name to use
        """
        self.input_db_path = input_db_path
        self.output_db_path = output_db_path
        self.model_name = model_name
        
        # Configure Gemini API
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel(model_name)
            print(f"✓ Gemini API configured with model: {model_name}")
        else:
            raise ValueError("Gemini API key is required!")
        
        # Copy input database to output database
        self._prepare_output_database()
    
    def _prepare_output_database(self):
        """Copy input database and add evaluation columns."""
        # Copy the database
        shutil.copy2(self.input_db_path, self.output_db_path)
        print(f"✓ Database copied to: {self.output_db_path}")
        
        # Add new columns for evaluation scores
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            # Add evaluation columns
            new_columns = [
                "ai_answer_score INTEGER",
                "article_1_score INTEGER",
                "article_2_score INTEGER",
                "article_3_score INTEGER",
                "article_4_score INTEGER",
                "article_5_score INTEGER",
                "article_6_score INTEGER",
                "article_7_score INTEGER",
                "article_8_score INTEGER",
                "evaluation_reasoning TEXT",
                "evaluation_status TEXT DEFAULT 'pending'",
                "evaluated_at TIMESTAMP"
            ]
            
            for column in new_columns:
                try:
                    cursor.execute(f"ALTER TABLE qa_faiss_results ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    # Column already exists
                    pass
            
            conn.commit()
            print("✓ Evaluation columns added to database")
    
    def truncate_text(self, text: str, max_length: int = 2000) -> str:
        """Truncate text to specified length."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def build_evaluation_prompt(self, 
                                question: str,
                                original_answer: str,
                                ai_generated_answer: str,
                                articles: List[Dict[str, str]]) -> str:
        """
        Build the prompt for Gemini to evaluate the QA results.
        
        Args:
            question: Original question
            original_answer: Ground truth answer
            ai_generated_answer: AI-generated answer from Ollama
            articles: List of article snippets (truncated to 2000 chars each)
            
        Returns:
            Formatted prompt string
        """
        # Build article snippets section
        article_sections = []
        for i, article in enumerate(articles, 1):
            snippet = self.truncate_text(article.get('snippet', ''), 2000)
            if snippet:
                article_sections.append(f"""
**Article {i}**
Title: {article.get('title', 'N/A')}
Snippet: {snippet}
""")
        
        articles_text = "\n".join(article_sections)
        
        prompt = f"""You are an expert evaluator assessing the quality of AI-generated answers and source materials for question-answering tasks.

**QUESTION:**
{question}

**ORIGINAL/GROUND TRUTH ANSWER:**
{original_answer}

**AI-GENERATED ANSWER (to be evaluated):**
{ai_generated_answer}

**SOURCE ARTICLES (used to generate the AI answer):**
{articles_text}

**YOUR TASK:**
Please evaluate the following on a Likert scale of 1-5:

1. **AI Answer Quality Score (1-5):** How well does the AI-generated answer match the original answer in terms of:
   - Factual accuracy
   - Completeness of information
   - Relevance to the question
   - Overall quality
   
   Scale:
   - 1: Very poor, incorrect or irrelevant
   - 2: Poor, mostly incorrect or incomplete
   - 3: Fair, partially correct with gaps
   - 4: Good, mostly correct and complete
   - 5: Excellent, highly accurate and comprehensive

2. **Article Quality Scores (1-5 for each article):** For each of the 8 articles, evaluate how useful and relevant the snippet is for answering the question:
   
   Scale:
   - 1: Not relevant or useful at all
   - 2: Slightly relevant but not helpful
   - 3: Moderately relevant and somewhat helpful
   - 4: Quite relevant and helpful
   - 5: Highly relevant and very helpful
   - 0: No content/empty snippet

**IMPORTANT:** Respond in VALID JSON format only (no markdown code blocks, no extra text):

{{
  "ai_answer_score": <1-5>,
  "article_scores": [<score1>, <score2>, <score3>, <score4>, <score5>, <score6>, <score7>, <score8>],
  "reasoning": "Brief explanation of your evaluation (2-3 sentences)"
}}

Provide your evaluation now:"""
        
        return prompt
    
    def parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's JSON response.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Dictionary with parsed scores
        """
        try:
            # Remove markdown code blocks if present
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            # Parse JSON
            data = json.loads(text)
            
            # Validate structure
            if "ai_answer_score" not in data or "article_scores" not in data:
                raise ValueError("Missing required fields in response")
            
            if len(data["article_scores"]) != 8:
                raise ValueError(f"Expected 8 article scores, got {len(data['article_scores'])}")
            
            return {
                'ai_answer_score': int(data['ai_answer_score']),
                'article_scores': [int(score) for score in data['article_scores']],
                'reasoning': data.get('reasoning', ''),
                'status': 'success'
            }
            
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parsing error: {e}")
            print(f"  Raw response: {response_text[:200]}...")
            return {'status': 'error', 'error': f'JSON parsing failed: {e}'}
        except Exception as e:
            print(f"  ⚠ Parsing error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def evaluate_single_entry(self, entry_id: int) -> Dict[str, Any]:
        """
        Evaluate a single QA entry using Gemini.
        
        Args:
            entry_id: ID of the entry to evaluate
            
        Returns:
            Evaluation result dictionary
        """
        print(f"\n{'='*80}")
        print(f"[ID {entry_id}] Evaluating entry...")
        
        try:
            # Load entry from database
            with sqlite3.connect(self.output_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT question, answer, ollama_generated_answer,
                           article_1_title, article_1_snippet,
                           article_2_title, article_2_snippet,
                           article_3_title, article_3_snippet,
                           article_4_title, article_4_snippet,
                           article_5_title, article_5_snippet,
                           article_6_title, article_6_snippet,
                           article_7_title, article_7_snippet,
                           article_8_title, article_8_snippet
                    FROM qa_faiss_results
                    WHERE id = ? AND processing_status = 'success'
                ''', (entry_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    return {
                        'id': entry_id,
                        'status': 'error',
                        'error': 'Entry not found or not successfully processed'
                    }
                
                question = row[0]
                original_answer = row[1]
                ai_answer = row[2]
                
                # Parse articles
                articles = []
                for i in range(8):
                    base_idx = 3 + (i * 2)
                    articles.append({
                        'title': row[base_idx],
                        'snippet': row[base_idx + 1]
                    })
                
                print(f"Q: {question[:100]}...")
            
            # Build evaluation prompt
            prompt = self.build_evaluation_prompt(
                question, original_answer, ai_answer, articles
            )
            
            # Call Gemini API
            print(f"  🤖 Calling Gemini API for evaluation...")
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            print(f"  ✓ Received response ({len(response_text)} chars)")
            
            # Parse response
            evaluation = self.parse_gemini_response(response_text)
            
            if evaluation['status'] == 'success':
                print(f"  📊 AI Answer Score: {evaluation['ai_answer_score']}/5")
                print(f"  📊 Article Scores: {evaluation['article_scores']}")
                
                return {
                    'id': entry_id,
                    'status': 'success',
                    'ai_answer_score': evaluation['ai_answer_score'],
                    'article_scores': evaluation['article_scores'],
                    'reasoning': evaluation['reasoning']
                }
            else:
                return {
                    'id': entry_id,
                    'status': 'error',
                    'error': evaluation.get('error', 'Unknown error')
                }
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return {
                'id': entry_id,
                'status': 'error',
                'error': str(e)
            }
    
    def store_evaluation(self, result: Dict[str, Any]):
        """Store evaluation result in database."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            if result['status'] == 'error':
                cursor.execute('''
                    UPDATE qa_faiss_results
                    SET evaluation_status = 'error',
                        evaluation_reasoning = ?,
                        evaluated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (result['error'], result['id']))
            else:
                cursor.execute('''
                    UPDATE qa_faiss_results
                    SET ai_answer_score = ?,
                        article_1_score = ?,
                        article_2_score = ?,
                        article_3_score = ?,
                        article_4_score = ?,
                        article_5_score = ?,
                        article_6_score = ?,
                        article_7_score = ?,
                        article_8_score = ?,
                        evaluation_reasoning = ?,
                        evaluation_status = 'success',
                        evaluated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    result['ai_answer_score'],
                    *result['article_scores'],
                    result['reasoning'],
                    result['id']
                ))
            
            conn.commit()
    
    def get_entries_to_evaluate(self) -> List[int]:
        """Get all entry IDs that need evaluation."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM qa_faiss_results
                WHERE processing_status = 'success'
                AND (evaluation_status IS NULL OR evaluation_status = 'pending')
                ORDER BY id
            ''')
            return [row[0] for row in cursor.fetchall()]
    
    def get_evaluation_progress(self) -> Dict[str, Any]:
        """Get evaluation progress statistics."""
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM qa_faiss_results 
                WHERE processing_status = 'success'
            ''')
            total = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM qa_faiss_results 
                WHERE evaluation_status = 'success'
            ''')
            evaluated = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM qa_faiss_results 
                WHERE evaluation_status = 'error'
            ''')
            errors = cursor.fetchone()[0]
            
            return {
                'total': total,
                'evaluated': evaluated,
                'errors': errors,
                'pending': total - evaluated - errors,
                'progress_percent': (evaluated / total * 100) if total > 0 else 0
            }
    
    def run_evaluation(self, delay_between_requests: float = 2.0):
        """
        Run complete evaluation pipeline.
        
        Args:
            delay_between_requests: Delay between API calls (to avoid rate limits)
        """
        print("🚀 Starting Gemini Evaluation")
        print("=" * 80)
        
        # Get entries to evaluate
        entry_ids = self.get_entries_to_evaluate()
        print(f"📚 Found {len(entry_ids)} entries to evaluate")
        
        if not entry_ids:
            print("✓ All entries already evaluated!")
            return
        
        print(f"\n🎯 Evaluating {len(entry_ids)} entries...")
        print("=" * 80)
        
        # Process each entry
        for i, entry_id in enumerate(entry_ids, 1):
            print(f"\n[{i}/{len(entry_ids)}] Progress: {i/len(entry_ids)*100:.1f}%")
            
            # Evaluate the entry
            result = self.evaluate_single_entry(entry_id)
            
            # Store the result
            self.store_evaluation(result)
            
            # Show progress
            progress = self.get_evaluation_progress()
            print(f"Overall: {progress['evaluated']} evaluated, {progress['errors']} errors, {progress['pending']} pending")
            
            # Delay between requests
            if i < len(entry_ids):
                time.sleep(delay_between_requests)
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final evaluation summary."""
        print("\n" + "=" * 80)
        print("🎉 EVALUATION COMPLETED")
        print("=" * 80)
        
        progress = self.get_evaluation_progress()
        
        print(f"📊 EVALUATION SUMMARY:")
        print(f"  • Total Entries: {progress['total']}")
        print(f"  • Successfully Evaluated: {progress['evaluated']}")
        print(f"  • Errors: {progress['errors']}")
        print(f"  • Success Rate: {progress['evaluated']/progress['total']*100:.1f}%")
        
        # Get average scores
        with sqlite3.connect(self.output_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(ai_answer_score) FROM qa_faiss_results 
                WHERE evaluation_status = 'success'
            ''')
            avg_ai_score = cursor.fetchone()[0] or 0
            
            article_scores = []
            for i in range(1, 9):
                cursor.execute(f'''
                    SELECT AVG(article_{i}_score) FROM qa_faiss_results 
                    WHERE evaluation_status = 'success'
                ''')
                article_scores.append(cursor.fetchone()[0] or 0)
            
            print(f"\n📈 AVERAGE SCORES:")
            print(f"  • AI Answer Score: {avg_ai_score:.2f}/5")
            print(f"  • Article Scores:")
            for i, score in enumerate(article_scores, 1):
                print(f"    - Article {i}: {score:.2f}/5")
        
        print(f"\n💾 Evaluated results saved to: {self.output_db_path}")


# Example usage
if __name__ == "__main__":
    import os
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("Example: export GEMINI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Initialize evaluator
    evaluator = QAGeminiEvaluator(
        input_db_path="qa_faiss_ollama_results.db",
        output_db_path="qa_faiss_ollama_evaluated.db",
        gemini_api_key=api_key,
        model_name="gemini-1.5-pro"
    )
    
    # Run evaluation with 2-second delay between requests
    evaluator.run_evaluation(delay_between_requests=2.0)