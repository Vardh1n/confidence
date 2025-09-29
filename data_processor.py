import pandas as pd
import re
import json
from typing import Dict, List, Optional, Tuple, Set, Iterator
from data_store import QADataStore
from classifier import ModernTextClassifier
from tqdm import tqdm  # Add this import for progress bar

class NaturalQuestionsProcessor:
    def __init__(self, db_path: str = "qa_database.db", chunk_size: int = 1000, save_frequency: int = 100):
        """
        Initialize the processor with database and classifier.
        
        Args:
            db_path: Path to the SQLite database
            chunk_size: Number of entries to process in each chunk
            save_frequency: Save to database every N entries
        """
        self.db = QADataStore(db_path)
        self.classifier = ModernTextClassifier(similarity_threshold=0.6)
        self.chunk_size = chunk_size
        self.save_frequency = save_frequency
        self._processed_questions = None  # Cache for processed questions
        self._processed_ids = None  # Cache for processed IDs
        
    def _get_processed_questions(self) -> Set[str]:
        """Get set of already processed questions for duplicate checking."""
        if self._processed_questions is None:
            print("Loading existing questions from database for duplicate checking...")
            existing_entries = self.db.get_all_entries()
            self._processed_questions = {entry['question'] for entry in existing_entries}
            print(f"Found {len(self._processed_questions)} existing questions in database")
        return self._processed_questions
    
    def _get_processed_ids(self) -> Set[str]:
        """Get set of already processed entry IDs for duplicate checking."""
        if self._processed_ids is None:
            print("Loading existing entry IDs from database for duplicate checking...")
            existing_entries = self.db.get_all_entries()
            self._processed_ids = {str(entry['id']) for entry in existing_entries}
            print(f"Found {len(self._processed_ids)} existing entry IDs in database")
        return self._processed_ids
    
    def _is_duplicate_question(self, question: str) -> bool:
        """Check if question has already been processed."""
        processed_questions = self._get_processed_questions()
        return question in processed_questions
    
    def _is_duplicate_id(self, entry_id: str) -> bool:
        """Check if entry ID has already been processed."""
        processed_ids = self._get_processed_ids()
        return str(entry_id) in processed_ids
    
    def _add_to_processed_cache(self, question: str, entry_id: str):
        """Add question and ID to processed cache."""
        if self._processed_questions is not None:
            self._processed_questions.add(question)
        if self._processed_ids is not None:
            self._processed_ids.add(str(entry_id))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text by removing HTML tags, underscores, hyphens, and extra whitespace."""
        if not text or pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', str(text))
        
        # Remove underscores and hyphens
        text = re.sub(r'[_\-]', ' ', text)
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation and normalize (keeping basic punctuation)
        text = re.sub(r'[^\w\s\?\.\!\,\:\;]', '', text)
        
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
        Load JSONL file and return list of entries (for small datasets or testing).
        
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

    def load_jsonl_chunks(self, file_path: str, chunk_size: Optional[int] = None) -> Iterator[List[Dict]]:
        """
        Load JSONL file in chunks to handle large datasets efficiently.
        
        Args:
            file_path: Path to the JSONL file
            chunk_size: Number of entries per chunk (uses instance default if None)
            
        Yields:
            Lists of JSON entries (chunks)
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        chunk = []
        total_loaded = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for idx, line in enumerate(file):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            chunk.append(entry)
                            total_loaded += 1
                            
                            # Yield chunk when it reaches the desired size
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {idx + 1}: {e}")
                            continue
                            
                    if total_loaded % 10000 == 0 and total_loaded > 0:
                        print(f"Loaded {total_loaded} entries...")
                
                # Yield remaining entries in the last chunk
                if chunk:
                    yield chunk
                    
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
            return
        
        print(f"Successfully processed {total_loaded} entries from JSONL file")
    
    def process_entries(self, entries: List[Dict], skip_duplicates: bool = True) -> List[Dict]:
        """
        Process the JSONL entries and return processed entries.
        
        Args:
            entries: List of JSONL entries
            skip_duplicates: Whether to skip entries that are already processed
            
        Returns:
            List of processed entries
        """
        processed_entries = []
        skipped_count = 0
        saved_count = 0
        
        print(f"Processing {len(entries)} entries...")
        if skip_duplicates:
            print("Duplicate checking enabled - will skip already processed entries")
        
        # Create progress bar
        progress_bar = tqdm(total=len(entries), desc="Processing entries", unit="entry")
        
        for idx, entry in enumerate(entries):
            try:
                # Use example_id as our ID
                entry_id = entry.get('example_id', idx)
                
                # Check for duplicate ID first (fastest check)
                if skip_duplicates and self._is_duplicate_id(entry_id):
                    skipped_count += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'processed': len(processed_entries),
                        'saved': saved_count,
                        'skipped': skipped_count
                    })
                    continue
                
                # Extract and clean question
                question = self.parse_question(entry.get('question_text', ''))
                if not question:
                    progress_bar.update(1)
                    continue
                
                # Check for duplicate question (after cleaning)
                if skip_duplicates and self._is_duplicate_question(question):
                    skipped_count += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'processed': len(processed_entries),
                        'saved': saved_count,
                        'skipped': skipped_count
                    })
                    continue
                
                # Consolidate answers
                answer = self.consolidate_answers(entry)
                if not answer:
                    progress_bar.update(1)
                    continue
                
                # Classify the question to get category
                category = self.classifier.classify_sentence(question)
                
                # Create processed entry
                processed_entry = {
                    'id': entry_id,
                    'question': question,
                    'answer': answer,
                    'category': category
                }
                
                processed_entries.append(processed_entry)
                
                # Add to processed cache
                if skip_duplicates:
                    self._add_to_processed_cache(question, entry_id)
                
                # Save to database every save_frequency entries
                if len(processed_entries) % self.save_frequency == 0:
                    batch_saved = self.save_batch_to_database(processed_entries[-self.save_frequency:])
                    saved_count += batch_saved
                    
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'processed': len(processed_entries),
                    'saved': saved_count,
                    'skipped': skipped_count
                })
                    
            except Exception as e:
                print(f"Error processing entry {idx}: {e}")
                progress_bar.update(1)
                continue
        
        # Save any remaining entries
        remaining_entries = len(processed_entries) % self.save_frequency
        if remaining_entries > 0:
            batch_saved = self.save_batch_to_database(processed_entries[-remaining_entries:])
            saved_count += batch_saved
        
        progress_bar.close()
        
        if skip_duplicates and skipped_count > 0:
            print(f"Processing complete: {len(processed_entries)} new entries processed, {saved_count} saved to DB, {skipped_count} duplicates skipped")
        else:
            print(f"Processing complete: {len(processed_entries)} entries processed, {saved_count} saved to DB")
        
        return processed_entries

    def process_chunk(self, entries: List[Dict], skip_duplicates: bool = True) -> Tuple[List[Dict], int, int]:
        """
        Process a chunk of entries and return processed entries, skip count, and saved count.
        
        Args:
            entries: List of JSONL entries to process
            skip_duplicates: Whether to skip already processed entries
            
        Returns:
            Tuple of (processed_entries, skipped_count, saved_count)
        """
        processed_entries = []
        skipped_count = 0
        saved_count = 0
        
        # Create progress bar for this chunk
        progress_bar = tqdm(total=len(entries), desc="Processing chunk", unit="entry", leave=False)
        
        for idx, entry in enumerate(entries):
            try:
                # Use example_id as our ID
                entry_id = entry.get('example_id', f"chunk_{idx}")
                
                # Check for duplicate ID first (fastest check)
                if skip_duplicates and self._is_duplicate_id(entry_id):
                    skipped_count += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'processed': len(processed_entries),
                        'saved': saved_count,
                        'skipped': skipped_count
                    })
                    continue
                
                # Extract and clean question
                question = self.parse_question(entry.get('question_text', ''))
                if not question:
                    progress_bar.update(1)
                    continue
                
                # Check for duplicate question (after cleaning)
                if skip_duplicates and self._is_duplicate_question(question):
                    skipped_count += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'processed': len(processed_entries),
                        'saved': saved_count,
                        'skipped': skipped_count
                    })
                    continue
                
                # Consolidate answers
                answer = self.consolidate_answers(entry)
                if not answer:
                    progress_bar.update(1)
                    continue
                
                # Classify the question to get category
                category = self.classifier.classify_sentence(question)
                
                # Create processed entry
                processed_entry = {
                    'id': entry_id,
                    'question': question,
                    'answer': answer,
                    'category': category
                }
                
                processed_entries.append(processed_entry)
                
                # Add to processed cache
                if skip_duplicates:
                    self._add_to_processed_cache(question, entry_id)
                
                # Save to database every save_frequency entries
                if len(processed_entries) % self.save_frequency == 0:
                    batch_saved = self.save_batch_to_database(processed_entries[-self.save_frequency:])
                    saved_count += batch_saved
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'processed': len(processed_entries),
                    'saved': saved_count,
                    'skipped': skipped_count
                })
                    
            except Exception as e:
                print(f"Error processing entry {idx}: {e}")
                progress_bar.update(1)
                continue
        
        # Save any remaining entries
        remaining_entries = len(processed_entries) % self.save_frequency
        if remaining_entries > 0:
            batch_saved = self.save_batch_to_database(processed_entries[-remaining_entries:])
            saved_count += batch_saved
        
        progress_bar.close()
        
        return processed_entries, skipped_count, saved_count
    
    def save_batch_to_database(self, processed_entries: List[Dict]) -> int:
        """
        Save a batch of processed entries to the database.
        
        Args:
            processed_entries: List of processed entry dictionaries
            
        Returns:
            Number of entries successfully saved
        """
        saved_count = 0
        
        for entry in processed_entries:
            try:
                self.db.add_entry(
                    question=entry['question'],
                    answer=entry['answer'],
                    category=entry['category']
                )
                saved_count += 1
                    
            except Exception as e:
                print(f"Error saving entry: {e}")
                continue
        
        return saved_count
    
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

    def save_chunk_to_database(self, processed_entries: List[Dict]) -> int:
        """
        Save a chunk of processed entries to the database.
        
        Args:
            processed_entries: List of processed entry dictionaries
            
        Returns:
            Number of entries successfully saved
        """
        saved_count = 0
        
        for entry in processed_entries:
            try:
                self.db.add_entry(
                    question=entry['question'],
                    answer=entry['answer'],
                    category=entry['category']
                )
                saved_count += 1
                    
            except Exception as e:
                print(f"Error saving entry: {e}")
                continue
        
        return saved_count
    
    def process_and_save(self, entries: List[Dict], skip_duplicates: bool = True) -> Tuple[int, int, int]:
        """
        Process entries and save to database in one step.
        
        Args:
            entries: List of JSONL entries
            skip_duplicates: Whether to skip entries that are already processed
            
        Returns:
            Tuple of (processed_count, saved_count, skipped_count)
        """
        # Get initial counts for calculating skipped
        initial_db_count = len(self.db.get_all_entries())
        
        processed_entries = self.process_entries(entries, skip_duplicates)
        saved_count = self.save_to_database(processed_entries)
        
        # Calculate how many were skipped
        skipped_count = len(entries) - len(processed_entries)
        
        return len(processed_entries), saved_count, skipped_count

    def process_large_dataset(self, file_path: str, skip_duplicates: bool = True, 
                            limit: Optional[int] = None) -> Tuple[int, int, int]:
        """
        Process large datasets efficiently by streaming and chunking.
        
        Args:
            file_path: Path to the JSONL file
            skip_duplicates: Whether to skip already processed entries
            limit: Optional limit on total entries to process
            
        Returns:
            Tuple of (total_processed, total_saved, total_skipped)
        """
        total_processed = 0
        total_saved = 0
        total_skipped = 0
        entries_processed_count = 0
        
        print(f"Processing large dataset in chunks of {self.chunk_size}...")
        print(f"Saving to database every {self.save_frequency} entries...")
        if skip_duplicates:
            print("Duplicate checking enabled")
        
        # Get total number of lines for progress bar (approximate)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            print(f"Estimated total entries: {total_lines}")
        except:
            total_lines = None
            print("Could not estimate total entries")
        
        # Create overall progress bar
        overall_progress = tqdm(total=limit or total_lines, desc="Overall progress", unit="entry")
        
        try:
            for chunk_idx, chunk in enumerate(self.load_jsonl_chunks(file_path)):
                if limit and entries_processed_count >= limit:
                    break
                
                # Limit chunk size if we're near the limit
                if limit:
                    remaining = limit - entries_processed_count
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                
                print(f"\nProcessing chunk {chunk_idx + 1} ({len(chunk)} entries)...")
                
                # Process chunk
                processed_entries, skipped_count, saved_count = self.process_chunk(chunk, skip_duplicates)
                
                # Update totals
                total_processed += len(processed_entries)
                total_saved += saved_count
                total_skipped += skipped_count
                entries_processed_count += len(chunk)
                
                # Update overall progress bar
                overall_progress.update(len(chunk))
                overall_progress.set_postfix({
                    'processed': total_processed,
                    'saved': total_saved,
                    'skipped': total_skipped
                })
                
                print(f"Chunk {chunk_idx + 1} complete: {len(processed_entries)} processed, "
                      f"{saved_count} saved, {skipped_count} skipped")
                print(f"Total so far: {total_processed} processed, {total_saved} saved, {total_skipped} skipped")
                print("-" * 50)
                
        except KeyboardInterrupt:
            print(f"\nProcessing interrupted by user")
            print(f"Data saved up to this point: {total_saved} entries")
        except Exception as e:
            print(f"Error during processing: {e}")
            print(f"Data saved up to this point: {total_saved} entries")
        finally:
            overall_progress.close()
        
        return total_processed, total_saved, total_skipped
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about the processed data."""
        stats = {
            'total_entries': len(self.db.get_all_entries()),
            'categories': self.db.get_all_categories(),
            'category_stats': self.db.get_category_stats()
        }
        return stats
    
    def reset_processed_cache(self):
        """Reset the processed questions/IDs cache to force reload from database."""
        self._processed_questions = None
        self._processed_ids = None
    
    def get_resume_info(self) -> Dict:
        """Get information about current processing state for resuming."""
        existing_entries = self.db.get_all_entries()
        if not existing_entries:
            return {
                'total_processed': 0,
                'last_processed_id': None,
                'can_resume': False
            }
        
        # Find the highest processed ID (assuming IDs are sequential)
        processed_ids = [entry['id'] for entry in existing_entries if entry['id'] is not None]
        if processed_ids:
            try:
                numeric_ids = [int(id) for id in processed_ids if str(id).isdigit()]
                last_id = max(numeric_ids) if numeric_ids else None
            except:
                last_id = None
        else:
            last_id = None
        
        return {
            'total_processed': len(existing_entries),
            'last_processed_id': last_id,
            'can_resume': True
        }

def load_and_process_dataset(file_path: str, limit: Optional[int] = None, skip_duplicates: bool = True):
    """
    Main function to load and process the Natural Questions JSONL dataset with resume capability.
    For small datasets or testing purposes.
    
    Args:
        file_path: Path to the JSONL file
        limit: Optional limit on number of entries to process
        skip_duplicates: Whether to skip already processed entries (enables resume)
    """
    print("Loading JSONL dataset...")
    
    # Initialize processor
    processor = NaturalQuestionsProcessor()
    
    # Show resume information
    if skip_duplicates:
        resume_info = processor.get_resume_info()
        print(f"Resume info: {resume_info['total_processed']} entries already processed")
        if resume_info['last_processed_id']:
            print(f"Last processed ID: {resume_info['last_processed_id']}")
    
    # Load JSONL entries
    entries = processor.load_jsonl(file_path, limit)
    
    if not entries:
        print("No entries loaded. Exiting.")
        return
    
    # Process and save
    print(f"\nStarting processing (skip_duplicates={skip_duplicates})...")
    processed_count, saved_count, skipped_count = processor.process_and_save(entries, skip_duplicates)
    
    # Print statistics
    print(f"\nProcessing completed!")
    print(f"Total entries in file: {len(entries)}")
    print(f"New entries processed: {processed_count}")
    print(f"Entries saved: {saved_count}")
    if skip_duplicates:
        print(f"Duplicate entries skipped: {skipped_count}")
    
    stats = processor.get_processing_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total entries in DB: {stats['total_entries']}")
    print(f"Categories found: {len(stats['categories'])}")
    print(f"Top categories:")
    for category, count in stats['category_stats'][:10]:
        print(f"  {category}: {count} entries")

def load_and_process_large_dataset(file_path: str, chunk_size: int = 2000, 
                                 limit: Optional[int] = None, skip_duplicates: bool = True):
    """
    Main function to efficiently process large JSONL datasets (like 15GB files).
    
    Args:
        file_path: Path to the JSONL file
        chunk_size: Number of entries to process in each chunk
        limit: Optional limit on total entries to process
        skip_duplicates: Whether to skip already processed entries
    """
    print(f"Processing large dataset: {file_path}")
    print(f"Chunk size: {chunk_size}")
    
    # Initialize processor with chunk size
    processor = NaturalQuestionsProcessor(chunk_size=chunk_size)
    
    # Show resume information
    if skip_duplicates:
        resume_info = processor.get_resume_info()
        print(f"Resume info: {resume_info['total_processed']} entries already processed")
        if resume_info['last_processed_id']:
            print(f"Last processed ID: {resume_info['last_processed_id']}")
    
    # Process the dataset
    processed_count, saved_count, skipped_count = processor.process_large_dataset(
        file_path, skip_duplicates, limit
    )
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total entries processed: {processed_count}")
    print(f"Total entries saved: {saved_count}")
    if skip_duplicates:
        print(f"Total duplicates skipped: {skipped_count}")
    
    # Show database statistics
    stats = processor.get_processing_stats()
    print(f"\nFinal Database Statistics:")
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
    # For testing with small datasets
    #file_path = "dataset.jsonl"  # Replace with your actual file path
    
    # First, test with a single entry
    large_file_path = "dataset.jsonl"  # Your 15GB file
    print("Testing with sample entry:")
    process_sample_entry(large_file_path)
    
    print("\n" + "="*50 + "\n")
    
    # For small datasets or testing (loads everything into memory)
    #print("Processing small dataset:")
    #load_and_process_dataset(file_path, limit=500, skip_duplicates=True)
    
    print("\n" + "="*50 + "\n")
    
    # For large datasets like 15GB files (memory-efficient chunked processing)
    print("Processing large dataset with chunked approach:")
    
    # Adjust chunk_size based on your available RAM:
    # - 1000 entries ≈ 50-100MB RAM
    # - 2000 entries ≈ 100-200MB RAM  
    # - 5000 entries ≈ 250-500MB RAM
    load_and_process_large_dataset(
        file_path=large_file_path,
        chunk_size=10000,  # Adjust based on your system's RAM
        skip_duplicates=True  # Enable resume capability
    )
    
    # To process the entire 15GB dataset:
    # load_and_process_large_dataset(large_file_path, chunk_size=2000, skip_duplicates=True)
    
    # To test with a limited number of entries:
    # load_and_process_large_dataset(large_file_path, chunk_size=2000, limit=10000, skip_duplicates=True)