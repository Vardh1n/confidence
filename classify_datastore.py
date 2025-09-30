import os
import sys
from typing import List, Dict
import sqlite3
from data_store import QADataStore
from advanced_classifier import AdvancedTextClassifier

class DatastoreClassifier:
    def __init__(self, 
                 db_path: str = "qa_database.db",
                 model_name: str = 'microsoft/DialoGPT-medium',
                 similarity_threshold: float = 0.75,
                 merge_threshold: float = 0.85):
        """
        Initialize the datastore classifier.
        
        Args:
            db_path: Path to the SQLite database
            model_name: HuggingFace model name for embeddings
            similarity_threshold: Minimum similarity to assign to existing category
            merge_threshold: Threshold for automatically merging similar categories
        """
        self.db_path = db_path
        self.datastore = QADataStore(db_path)
        self.classifier = AdvancedTextClassifier(
            model_name=model_name,
            similarity_threshold=similarity_threshold,
            merge_threshold=merge_threshold
        )
        
        # Add the 'class' column to the database if it doesn't exist
        self._add_class_column()
    
    def _add_class_column(self):
        """Add a 'class' column to the qa_entries table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if 'class' column exists
            cursor.execute("PRAGMA table_info(qa_entries)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'class' not in columns:
                print("Adding 'class' column to qa_entries table...")
                cursor.execute('ALTER TABLE qa_entries ADD COLUMN class TEXT')
                conn.commit()
                print("Successfully added 'class' column.")
            else:
                print("'class' column already exists.")
    
    def classify_all_questions(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Classify all questions in the datastore and update the 'class' column.
        
        Args:
            batch_size: Number of entries to process at a time
            
        Returns:
            Dictionary with classification statistics
        """
        # Get all entries from the datastore
        all_entries = self.datastore.get_all_entries()
        total_entries = len(all_entries)
        
        if total_entries == 0:
            print("No entries found in the datastore.")
            return {"total": 0, "classified": 0, "updated": 0}
        
        print(f"Found {total_entries} entries to classify...")
        
        classified_count = 0
        updated_count = 0
        classification_stats = {}
        
        # Process entries in batches
        for i in range(0, total_entries, batch_size):
            batch = all_entries[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(total_entries-1)//batch_size + 1} "
                  f"(entries {i+1}-{min(i+batch_size, total_entries)})...")
            
            for entry in batch:
                entry_id = entry['id']
                question = entry['question']
                current_class = entry.get('class')
                
                # Skip if already classified (unless you want to re-classify)
                if current_class is not None and current_class.strip():
                    print(f"  Entry {entry_id}: Already classified as '{current_class}' - skipping")
                    continue
                
                try:
                    # Classify the question
                    predicted_class = self.classifier.classify_question(question)
                    
                    # Update the database with the classification
                    success = self._update_entry_class(entry_id, predicted_class)
                    
                    if success:
                        classified_count += 1
                        updated_count += 1
                        
                        # Track classification statistics
                        if predicted_class in classification_stats:
                            classification_stats[predicted_class] += 1
                        else:
                            classification_stats[predicted_class] = 1
                        
                        print(f"  Entry {entry_id}: Classified as '{predicted_class}'")
                    else:
                        print(f"  Entry {entry_id}: Failed to update database")
                
                except Exception as e:
                    print(f"  Entry {entry_id}: Error during classification - {str(e)}")
                    continue
        
        # Display final statistics
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total entries processed: {total_entries}")
        print(f"Entries classified: {classified_count}")
        print(f"Database updates: {updated_count}")
        
        print(f"\nClassification Distribution:")
        for class_name, count in sorted(classification_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} questions")
        
        # Display classifier summary
        print(f"\nFinal Classifier State:")
        self.classifier.display_category_summary()
        
        return {
            "total": total_entries,
            "classified": classified_count,
            "updated": updated_count,
            "distribution": classification_stats
        }
    
    def _update_entry_class(self, entry_id: int, classification: str) -> bool:
        """
        Update the 'class' column for a specific entry.
        
        Args:
            entry_id: The ID of the entry to update
            classification: The classification to assign
            
        Returns:
            True if successful, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE qa_entries SET class = ? WHERE id = ?',
                (classification, entry_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def classify_new_question(self, question: str, answer: str) -> str:
        """
        Classify a new question and add it to the datastore with its classification.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            The assigned classification
        """
        # Classify the question
        classification = self.classifier.classify_question(question)
        
        # Add to datastore with classification
        entry_id = self.datastore.add_entry(question, answer, classification)
        
        print(f"Added new entry {entry_id} with classification '{classification}'")
        return classification
    
    def reclassify_category(self, target_class: str) -> int:
        """
        Re-classify all questions currently assigned to a specific class.
        
        Args:
            target_class: The class to re-classify
            
        Returns:
            Number of entries re-classified
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM qa_entries WHERE class = ?', (target_class,))
            entries = [dict(row) for row in cursor.fetchall()]
        
        if not entries:
            print(f"No entries found with class '{target_class}'")
            return 0
        
        print(f"Re-classifying {len(entries)} entries from class '{target_class}'...")
        
        reclassified_count = 0
        for entry in entries:
            try:
                new_class = self.classifier.classify_question(entry['question'])
                if self._update_entry_class(entry['id'], new_class):
                    reclassified_count += 1
                    print(f"  Entry {entry['id']}: '{target_class}' -> '{new_class}'")
            except Exception as e:
                print(f"  Entry {entry['id']}: Error - {str(e)}")
        
        return reclassified_count
    
    def get_classification_stats(self) -> List[tuple]:
        """
        Get statistics about classifications in the database.
        
        Returns:
            List of tuples (class, count)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT class, COUNT(*) as count 
                FROM qa_entries 
                WHERE class IS NOT NULL 
                GROUP BY class 
                ORDER BY count DESC
            ''')
            return cursor.fetchall()
    
    def export_classified_data(self, filepath: str):
        """Export all classified data to a JSON file."""
        import json
        
        entries = self.datastore.get_all_entries()
        with open(filepath, 'w') as f:
            json.dump(entries, f, indent=2, default=str)
        
        print(f"Exported {len(entries)} entries to {filepath}")
    
    def save_classifier_state(self, filepath: str):
        """Save the classifier model state."""
        self.classifier.save_model_state(filepath)
    
    def load_classifier_state(self, filepath: str):
        """Load a pre-trained classifier state."""
        self.classifier.load_model_state(filepath)


def main():
    """Main function to run the classification process."""
    print("Q&A Datastore Classifier")
    print("=" * 40)
    
    # Check if database exists
    db_path = "qa_database.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found. Please ensure the database exists with Q&A entries.")
        return
    
    # Initialize the classifier
    print("Initializing classifier...")
    try:
        classifier = DatastoreClassifier(
            db_path=db_path,
            similarity_threshold=0.75,
            merge_threshold=0.85
        )
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return
    
    # Get current state
    datastore = QADataStore(db_path)
    total_entries = len(datastore.get_all_entries())
    print(f"Found {total_entries} entries in the database.")
    
    if total_entries == 0:
        print("No entries to classify. Exiting.")
        return
    
    # Show current classification stats
    current_stats = classifier.get_classification_stats()
    if current_stats:
        print(f"\nCurrent classification stats:")
        for class_name, count in current_stats:
            print(f"  {class_name}: {count} entries")
    else:
        print("\nNo existing classifications found.")
    
    # Ask user what to do
    print(f"\nOptions:")
    print("1. Classify all unclassified questions")
    print("2. Re-classify all questions (overwrites existing classifications)")
    print("3. Re-classify specific category")
    print("4. Export classified data")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\nClassifying all unclassified questions...")
        stats = classifier.classify_all_questions()
        
    elif choice == "2":
        confirm = input("This will re-classify ALL questions. Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            # Clear existing classifications
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE qa_entries SET class = NULL')
                conn.commit()
            
            print("\nRe-classifying all questions...")
            stats = classifier.classify_all_questions()
        else:
            print("Operation cancelled.")
            return
            
    elif choice == "3":
        if not current_stats:
            print("No existing classifications to re-classify.")
            return
            
        print(f"\nCurrent classes:")
        for i, (class_name, count) in enumerate(current_stats, 1):
            print(f"{i}. {class_name} ({count} entries)")
        
        try:
            class_idx = int(input(f"\nSelect class to re-classify (1-{len(current_stats)}): ")) - 1
            if 0 <= class_idx < len(current_stats):
                target_class = current_stats[class_idx][0]
                count = classifier.reclassify_category(target_class)
                print(f"Re-classified {count} entries.")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
            
    elif choice == "4":
        output_file = input("Enter output filename (default: classified_qa_data.json): ").strip()
        if not output_file:
            output_file = "classified_qa_data.json"
        
        classifier.export_classified_data(output_file)
        
        # Also save classifier state
        classifier_file = "classifier_state.json"
        classifier.save_classifier_state(classifier_file)
        print(f"Classifier state saved to {classifier_file}")
        
    elif choice == "5":
        print("Exiting.")
        return
        
    else:
        print("Invalid choice.")
        return
    
    # Final statistics
    print(f"\nFinal classification statistics:")
    final_stats = classifier.get_classification_stats()
    for class_name, count in final_stats:
        print(f"  {class_name}: {count} entries")


if __name__ == "__main__":
    main()