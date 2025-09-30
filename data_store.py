import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class QADataStore:
    def __init__(self, db_path: str = "qa_database.db"):
        """
        Initialize the Q&A datastore with SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create the table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='qa_entries'
            ''')
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Create new table with all columns
                cursor.execute('''
                    CREATE TABLE qa_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        category TEXT NOT NULL
                    )
                ''')
            else:
                # Check if category column exists in existing table
                cursor.execute("PRAGMA table_info(qa_entries)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'category' not in columns:
                    # Add category column if it doesn't exist
                    cursor.execute('''
                        ALTER TABLE qa_entries 
                        ADD COLUMN category TEXT DEFAULT ''
                    ''')
        
            # Create indexes (only create category index if column exists)
            cursor.execute("PRAGMA table_info(qa_entries)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'category' in columns:
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON qa_entries(category)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question ON qa_entries(question)')
            
            conn.commit()
    
    def add_entry(self, question: str, answer: str, category: str) -> int:
        """
        Add a new Q&A entry to the database.
        
        Args:
            question: The question text
            answer: The answer text
            category: The category for this Q&A pair
            
        Returns:
            The ID of the newly created entry
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO qa_entries (question, answer, category)
                VALUES (?, ?, ?)
            ''', (question, answer, category))
            
            entry_id = cursor.lastrowid
            conn.commit()
            return entry_id
    
    def get_entry_by_id(self, entry_id: int) -> Optional[Dict]:
        """
        Get a specific entry by its ID.
        
        Args:
            entry_id: The ID of the entry to retrieve
            
        Returns:
            Dictionary with entry data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM qa_entries WHERE id = ?', (entry_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_entries_by_category(self, category: str) -> List[Dict]:
        """
        Get all entries for a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of dictionaries with entry data
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM qa_entries WHERE category = ? ORDER BY created_at DESC', (category,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def search_questions(self, search_term: str) -> List[Dict]:
        """
        Search for entries containing the search term in the question.
        
        Args:
            search_term: The term to search for
            
        Returns:
            List of dictionaries with matching entries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM qa_entries 
                WHERE question LIKE ? OR answer LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{search_term}%', f'%{search_term}%'))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_all_entries(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all entries from the database.
        
        Args:
            limit: Optional limit on number of entries to return
            
        Returns:
            List of dictionaries with entry data
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM qa_entries ORDER BY created_at DESC'
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_all_categories(self) -> List[str]:
        """
        Get all unique categories in the database.
        
        Returns:
            List of category names
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT category FROM qa_entries ORDER BY category')
            
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    
    def update_entry(self, entry_id: int, question: str = None, answer: str = None, category: str = None) -> bool:
        """
        Update an existing entry.
        
        Args:
            entry_id: The ID of the entry to update
            question: New question text (optional)
            answer: New answer text (optional)
            category: New category (optional)
            
        Returns:
            True if entry was updated, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query
            updates = []
            params = []
            
            if question is not None:
                updates.append('question = ?')
                params.append(question)
            
            if answer is not None:
                updates.append('answer = ?')
                params.append(answer)
            
            if category is not None:
                updates.append('category = ?')
                params.append(category)
            
            if not updates:
                return False
            
            updates.append('updated_at = CURRENT_TIMESTAMP')
            params.append(entry_id)
            
            query = f'UPDATE qa_entries SET {", ".join(updates)} WHERE id = ?'
            cursor.execute(query, params)
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_entry(self, entry_id: int) -> bool:
        """
        Delete an entry by ID.
        
        Args:
            entry_id: The ID of the entry to delete
            
        Returns:
            True if entry was deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM qa_entries WHERE id = ?', (entry_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_category_stats(self) -> List[Tuple[str, int]]:
        """
        Get statistics about entries per category.
        
        Returns:
            List of tuples (category, count)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM qa_entries 
                GROUP BY category 
                ORDER BY count DESC
            ''')
            
            return cursor.fetchall()
    
    def export_to_json(self, filepath: str):
        """Export all data to a JSON file."""
        import json
        
        entries = self.get_all_entries()
        with open(filepath, 'w') as f:
            json.dump(entries, f, indent=2, default=str)
    
    def import_from_json(self, filepath: str):
        """Import data from a JSON file."""
        import json
        
        with open(filepath, 'r') as f:
            entries = json.load(f)
        
        for entry in entries:
            if all(key in entry for key in ['question', 'answer', 'category']):
                self.add_entry(entry['question'], entry['answer'], entry['category'])

    def remove_unused_columns(self):
        """
        Remove all columns except id, question, and answer from the qa_entries table.
        This will create a new table with only the essential columns and migrate the data.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create a new table with only the required columns
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_entries_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                )
            ''')
            
            # Copy data from old table to new table
            cursor.execute('''
                INSERT INTO qa_entries_new (id, question, answer)
                SELECT id, question, answer FROM qa_entries
            ''')
            
            # Drop the old table
            cursor.execute('DROP TABLE qa_entries')
            
            # Rename the new table to the original name
            cursor.execute('ALTER TABLE qa_entries_new RENAME TO qa_entries')
            
            # Recreate the index for question
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question ON qa_entries(question)')
            
            conn.commit()
            print("Successfully removed unused columns. Only id, question, and answer remain.")
    
    def add_category_column(self):
        """
        Add a new 'category' column to the qa_entries table and initialize it as empty.
        This will alter the existing table structure to include the category column.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # Add the category column with empty string as default
                cursor.execute('''
                    ALTER TABLE qa_entries 
                    ADD COLUMN category TEXT DEFAULT ''
                ''')
                
                # Create index for better query performance on the new column
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON qa_entries(category)')
                
                conn.commit()
                print("Successfully added 'category' column to qa_entries table.")
                
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print("Category column already exists in the table.")
                else:
                    print(f"Error adding category column: {e}")

    def close(self):
        """Close the database connection (for explicit cleanup if needed)."""
        # SQLite connections are closed automatically with context managers
        pass

# Example usage
if __name__ == "__main__":
    # Initialize the datastore
    db = QADataStore("qa_database.db")
    
    # Add some sample entries
    """    sample_entries = [
        ("What is Python?", "Python is a high-level programming language known for its simplicity and readability.", "programming"),
        ("How do you create a list in Python?", "You can create a list using square brackets: my_list = [1, 2, 3]", "programming"),
        ("What is machine learning?", "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.", "ai"),
        ("What is the capital of France?", "The capital of France is Paris.", "geography"),
        ("How do you make pasta?", "Boil water, add pasta, cook for 8-12 minutes, then drain and serve with sauce.", "cooking")
    ]
    
    print("Adding sample entries...")
    for question, answer, category in sample_entries:
        entry_id = db.add_entry(question, answer, category)
        print(f"Added entry {entry_id}: {question[:30]}...")
    

    # Display some statistics
    print(f"\nTotal entries: {len(db.get_all_entries())}")
    print(f"Categories: {db.get_all_categories()}")
    
    print("\nCategory statistics:")
    for category, count in db.get_category_stats():
        print(f"  {category}: {count} entries")
    
    # Search example
    print(f"\nSearching for 'Python':")
    results = db.search_questions("Python")
    for result in results:
        print(f"  Q: {result['question']}")
        print(f"  A: {result['answer'][:50]}...")
        print(f"  Category: {result['category']}\n")
    """
    #db.remove_unused_columns()
    #db.add_category_column()