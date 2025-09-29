import pandas as pd
import re
import json
from typing import Dict, List, Optional, Tuple
from data_store import QADataStore
from classifier import ModernTextClassifier

class NaturalQuestionsProcessor:
    def __init__(self, db_path: str = "qa_database.db"):
        """
        Initialize the processor with database and classifier.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db = QADataStore(db_path)
        self.classifier = ModernTextClassifier(similarity_threshold=0.6)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text by removing HTML tags and extra whitespace."""
        if not text or pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', str(text))
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation and normalize
        text = re.sub(r'[^\w\s\?\.\!\,\:\;\-]', '', text)
        
        return text.strip()
    
    def parse_question(self, question_text: str) -> str:
        """Parse and clean the question text."""
        if not question_text or pd.isna(question_text):
            return ""
        
        question = self.clean_text(question_text)
        
        # Ensure question ends with question mark if it doesn't have one
        if question and not question.endswith('?'):
            question += '?'
        
        return question
    
    def extract_answer_text(self, document_text: str, start_token: int, end_token: int) -> str:
        """
        Extract answer text from document using token positions.
        
        Args:
            document_text: The full document text
            start_token: Start token position
            end_token: End token position
            
        Returns:
            Extracted answer text
        """
        if not document_text or pd.isna(document_text):
            return ""
        
        # Split document into tokens (simple whitespace split)
        tokens = document_text.split()
        
        # Extract tokens within the range
        if 0 <= start_token < len(tokens) and start_token < end_token <= len(tokens):
            answer_tokens = tokens[start_token:end_token]
            answer = ' '.join(answer_tokens)
            return self.clean_text(answer)
        
        return ""
    
    def consolidate_answers(self, entry: Dict) -> str:
        """
        Consolidate short and long answers into one cohesive answer.
        
        Args:
            entry: Dictionary containing the JSONL entry data
            
        Returns:
            Consolidated answer text
        """
        document_text = entry.get('document_text', '')
        annotations = entry.get('annotations', [])
        
        if not annotations:
            return ""
        
        try:
            if not isinstance(annotations, list):
                return ""
            
            annotation = annotations[0]  # Take first annotation
            
            short_answers = annotation.get('short_answers', [])
            long_answer = annotation.get('long_answer', {})
            
            consolidated_answer = ""
            
            # Extract short answers
            short_answer_texts = []
            if short_answers:
                for short_ans in short_answers:
                    start_token = short_ans.get('start_token', 0)
                    end_token = short_ans.get('end_token', 0)
                    short_text = self.extract_answer_text(document_text, start_token, end_token)
                    if short_text:
                        short_answer_texts.append(short_text)
            
            # Extract long answer
            long_answer_text = ""
            if long_answer and 'start_token' in long_answer and 'end_token' in long_answer:
                start_token = long_answer['start_token']
                end_token = long_answer['end_token']
                long_answer_text = self.extract_answer_text(document_text, start_token, end_token)
            
            # Consolidate answers
            if short_answer_texts:
                # Use short answers as primary, with long answer as context
                primary_answer = "; ".join(short_answer_texts)
                if long_answer_text and len(long_answer_text) > len(primary_answer):
                    # If long answer is significantly longer, use it as additional context
                    consolidated_answer = f"{primary_answer}. {long_answer_text[:500]}..."
                else:
                    consolidated_answer = primary_answer
            elif long_answer_text:
                # Only long answer available
                consolidated_answer = long_answer_text[:500] + "..." if len(long_answer_text) > 500 else long_answer_text
            
            return consolidated_answer
            
        except Exception as e:
            print(f"Error processing annotations: {e}")
            return ""
    
    def load_jsonl(self, file_path: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Load JSONL file and return list of entries.
        
        Args:
            file_path: Path to the JSONL file
            limit: Optional limit on number of entries to load
            
        Returns:
            List of JSON entries
        """
        entries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for idx, line in enumerate(file):
                    if limit and idx >= limit:
                        break
                        
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {idx + 1}: {e}")
                            continue
                            
                if idx % 1000 == 0 and idx > 0:
                    print(f"Loaded {idx + 1} entries...")
                    
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
            return []
        
        print(f"Successfully loaded {len(entries)} entries from JSONL file")
        return entries
    
    def process_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Process the JSONL entries and return processed entries.
        
        Args:
            entries: List of JSONL entries
            
        Returns:
            List of processed entries
        """
        processed_entries = []
        
        print(f"Processing {len(entries)} entries...")
        
        for idx, entry in enumerate(entries):
            try:
                # Extract and clean question
                question = self.parse_question(entry.get('question_text', ''))
                if not question:
                    continue
                
                # Consolidate answers
                answer = self.consolidate_answers(entry)
                if not answer:
                    continue
                
                # Classify the question to get category
                category = self.classifier.classify_sentence(question)
                
                # Use example_id as our ID
                entry_id = entry.get('example_id', idx)
                
                # Create processed entry
                processed_entry = {
                    'id': entry_id,
                    'question': question,
                    'answer': answer,
                    'category': category
                }
                
                processed_entries.append(processed_entry)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} entries...")
                    
            except Exception as e:
                print(f"Error processing entry {idx}: {e}")
                continue
        
        return processed_entries
    
    def save_to_database(self, processed_entries: List[Dict]) -> int:
        """
        Save processed entries to the database.
        
        Args:
            processed_entries: List of processed entry dictionaries
            
        Returns:
            Number of entries successfully saved
        """
        saved_count = 0
        
        print(f"Saving {len(processed_entries)} entries to database...")
        
        for entry in processed_entries:
            try:
                self.db.add_entry(
                    question=entry['question'],
                    answer=entry['answer'],
                    category=entry['category']
                )
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"Saved {saved_count} entries...")
                    
            except Exception as e:
                print(f"Error saving entry: {e}")
                continue
        
        return saved_count
    
    def process_and_save(self, entries: List[Dict]) -> Tuple[int, int]:
        """
        Process entries and save to database in one step.
        
        Args:
            entries: List of JSONL entries
            
        Returns:
            Tuple of (processed_count, saved_count)
        """
        processed_entries = self.process_entries(entries)
        saved_count = self.save_to_database(processed_entries)
        
        return len(processed_entries), saved_count
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about the processed data."""
        stats = {
            'total_entries': len(self.db.get_all_entries()),
            'categories': self.db.get_all_categories(),
            'category_stats': self.db.get_category_stats()
        }
        return stats

def load_and_process_dataset(file_path: str, limit: Optional[int] = None):
    """
    Main function to load and process the Natural Questions JSONL dataset.
    
    Args:
        file_path: Path to the JSONL file
        limit: Optional limit on number of entries to process
    """
    print("Loading JSONL dataset...")
    
    # Initialize processor
    processor = NaturalQuestionsProcessor()
    
    # Load JSONL entries
    entries = processor.load_jsonl(file_path, limit)
    
    if not entries:
        print("No entries loaded. Exiting.")
        return
    
    # Process and save
    print("\nStarting processing...")
    processed_count, saved_count = processor.process_and_save(entries)
    
    # Print statistics
    print(f"\nProcessing completed!")
    print(f"Processed: {processed_count} entries")
    print(f"Saved: {saved_count} entries")
    
    stats = processor.get_processing_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total entries in DB: {stats['total_entries']}")
    print(f"Categories found: {len(stats['categories'])}")
    print(f"Top categories:")
    for category, count in stats['category_stats'][:10]:
        print(f"  {category}: {count} entries")

def process_sample_entry(file_path: str):
    """
    Process and display a single sample entry for testing.
    
    Args:
        file_path: Path to the JSONL file
    """
    processor = NaturalQuestionsProcessor()
    entries = processor.load_jsonl(file_path, limit=1)
    
    if entries:
        entry = entries[0]
        print("Sample entry structure:")
        print(f"Keys: {list(entry.keys())}")
        print(f"Question: {entry.get('question_text', 'N/A')}")
        print(f"Example ID: {entry.get('example_id', 'N/A')}")
        
        # Process the sample
        processed = processor.process_entries([entry])
        if processed:
            print(f"\nProcessed entry:")
            print(f"Question: {processed[0]['question']}")
            print(f"Answer: {processed[0]['answer'][:200]}...")
            print(f"Category: {processed[0]['category']}")

# Example usage
if __name__ == "__main__":
    # Process a sample of the dataset
    file_path = "dataset.jsonl"  # Replace with your actual file path
    
    # First, test with a single entry
    print("Testing with sample entry:")
    process_sample_entry(file_path)
    
    print("\n" + "="*50 + "\n")
    
    # Process only first 500 entries for testing
    load_and_process_dataset(file_path, limit=500)
    
    # To process the entire dataset, use:
    # load_and_process_dataset(file_path)