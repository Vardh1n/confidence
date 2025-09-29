import os
import sys
from typing import Optional, List, Dict
from data_store import QADataStore
from datetime import datetime

class QADatabaseVisualizer:
    def __init__(self, db_path: str = "qa_database.db"):
        """
        Initialize the database visualizer.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db = QADataStore(db_path)
        self.terminal_width = os.get_terminal_size().columns if hasattr(os, 'get_terminal_size') else 80
    
    def print_separator(self, char='-'):
        """Print a separator line."""
        print(char * self.terminal_width)
    
    def print_header(self, title: str):
        """Print a formatted header."""
        self.print_separator('=')
        padding = (self.terminal_width - len(title) - 2) // 2
        print(f"{'=' * padding} {title} {'=' * padding}")
        self.print_separator('=')
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to fit within specified length."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def format_entry(self, entry: Dict, show_full: bool = False) -> str:
        """Format a single database entry for display."""
        id_str = f"ID: {entry['id']}"
        category_str = f"Category: {entry['category']}"
        created_str = f"Created: {entry['created_at']}"
        
        if show_full:
            question = entry['question']
            answer = entry['answer']
        else:
            question = self.truncate_text(entry['question'], self.terminal_width - 15)
            answer = self.truncate_text(entry['answer'], self.terminal_width - 15)
        
        return f"""
{id_str:<15} {category_str:<20} {created_str}
Q: {question}
A: {answer}
"""
    
    def show_head(self, n: int = 5):
        """Show the first n entries."""
        self.print_header(f"HEAD - First {n} Entries")
        entries = self.db.get_all_entries(limit=n)
        
        if not entries:
            print("No entries found in the database.")
            return
        
        for i, entry in enumerate(entries, 1):
            print(f"\n[{i}]")
            print(self.format_entry(entry))
            if i < len(entries):
                self.print_separator()
    
    def show_tail(self, n: int = 5):
        """Show the last n entries."""
        self.print_header(f"TAIL - Last {n} Entries")
        all_entries = self.db.get_all_entries()
        
        if not all_entries:
            print("No entries found in the database.")
            return
        
        # Get last n entries (reverse order since get_all_entries returns newest first)
        tail_entries = all_entries[:n]
        tail_entries.reverse()  # Show oldest to newest
        
        for i, entry in enumerate(tail_entries, 1):
            print(f"\n[{i}]")
            print(self.format_entry(entry))
            if i < len(tail_entries):
                self.print_separator()
    
    def show_statistics(self):
        """Display comprehensive database statistics."""
        self.print_header("DATABASE STATISTICS")
        
        # Basic stats
        all_entries = self.db.get_all_entries()
        total_entries = len(all_entries)
        categories = self.db.get_all_categories()
        category_stats = self.db.get_category_stats()
        
        print(f"Total Entries: {total_entries}")
        print(f"Total Categories: {len(categories)}")
        print(f"Database File: {self.db.db_path}")
        
        if os.path.exists(self.db.db_path):
            file_size = os.path.getsize(self.db.db_path)
            print(f"Database Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
        # Category breakdown
        if category_stats:
            print(f"\nCategory Breakdown:")
            self.print_separator()
            print(f"{'Category':<20} {'Count':<10} {'Percentage':<10}")
            self.print_separator()
            
            for category, count in category_stats:
                percentage = (count / total_entries * 100) if total_entries > 0 else 0
                print(f"{category:<20} {count:<10} {percentage:6.1f}%")
        
        # Question and answer length statistics
        if all_entries:
            question_lengths = [len(entry['question']) for entry in all_entries]
            answer_lengths = [len(entry['answer']) for entry in all_entries]
            
            print(f"\nText Length Statistics:")
            self.print_separator()
            print(f"{'Metric':<25} {'Questions':<15} {'Answers':<15}")
            self.print_separator()
            print(f"{'Average Length':<25} {sum(question_lengths)/len(question_lengths):>10.1f} {sum(answer_lengths)/len(answer_lengths):>10.1f}")
            print(f"{'Minimum Length':<25} {min(question_lengths):>10} {min(answer_lengths):>10}")
            print(f"{'Maximum Length':<25} {max(question_lengths):>10} {max(answer_lengths):>10}")
        
        # Recent activity
        if all_entries:
            print(f"\nRecent Activity:")
            self.print_separator()
            recent_entries = all_entries[:3]
            for entry in recent_entries:
                created = entry['created_at']
                question_preview = self.truncate_text(entry['question'], 50)
                print(f"{created} - {entry['category']}: {question_preview}")
    
    def show_categories(self):
        """Display all categories with entry counts."""
        self.print_header("CATEGORIES")
        
        category_stats = self.db.get_category_stats()
        
        if not category_stats:
            print("No categories found.")
            return
        
        print(f"{'#':<3} {'Category':<25} {'Count':<10} {'Bar Chart'}")
        self.print_separator()
        
        max_count = max(count for _, count in category_stats) if category_stats else 1
        bar_width = min(50, self.terminal_width - 50)
        
        for i, (category, count) in enumerate(category_stats, 1):
            bar_length = int((count / max_count) * bar_width)
            bar = '█' * bar_length + '░' * (bar_width - bar_length)
            print(f"{i:<3} {category:<25} {count:<10} {bar}")
    
    def show_entry_details(self, entry_id: int):
        """Show full details of a specific entry."""
        entry = self.db.get_entry_by_id(entry_id)
        
        if not entry:
            print(f"Entry with ID {entry_id} not found.")
            return
        
        self.print_header(f"ENTRY DETAILS - ID {entry_id}")
        
        print(f"ID: {entry['id']}")
        print(f"Category: {entry['category']}")
        print(f"Created: {entry['created_at']}")
        print(f"Updated: {entry['updated_at']}")
        print()
        print("QUESTION:")
        print(entry['question'])
        print()
        print("ANSWER:")
        print(entry['answer'])
    
    def search_and_display(self, search_term: str, max_results: int = 10):
        """Search for entries and display results."""
        self.print_header(f"SEARCH RESULTS for '{search_term}'")
        
        results = self.db.search_questions(search_term)
        
        if not results:
            print(f"No entries found containing '{search_term}'.")
            return
        
        print(f"Found {len(results)} matching entries (showing first {max_results}):")
        print()
        
        for i, entry in enumerate(results[:max_results], 1):
            print(f"[{i}]")
            print(self.format_entry(entry))
            if i < min(len(results), max_results):
                self.print_separator()
        
        if len(results) > max_results:
            print(f"\n... and {len(results) - max_results} more results.")
    
    def show_category_entries(self, category: str, max_results: int = 10):
        """Show entries for a specific category."""
        self.print_header(f"CATEGORY: {category}")
        
        entries = self.db.get_entries_by_category(category)
        
        if not entries:
            print(f"No entries found for category '{category}'.")
            return
        
        print(f"Found {len(entries)} entries in category '{category}' (showing first {max_results}):")
        print()
        
        for i, entry in enumerate(entries[:max_results], 1):
            print(f"[{i}]")
            print(self.format_entry(entry))
            if i < min(len(entries), max_results):
                self.print_separator()
        
        if len(entries) > max_results:
            print(f"\n... and {len(entries) - max_results} more entries.")
    
    def interactive_menu(self):
        """Run an interactive menu for database exploration."""
        while True:
            self.print_header("QA DATABASE EXPLORER")
            print("Choose an option:")
            print("1. Show head (first 5 entries)")
            print("2. Show tail (last 5 entries)")
            print("3. Show statistics")
            print("4. Show categories")
            print("5. Show entry details by ID")
            print("6. Search entries")
            print("7. Show entries by category")
            print("8. Custom head/tail")
            print("9. Exit")
            print()
            
            choice = input("Enter your choice (1-9): ").strip()
            print()
            
            try:
                if choice == '1':
                    self.show_head()
                elif choice == '2':
                    self.show_tail()
                elif choice == '3':
                    self.show_statistics()
                elif choice == '4':
                    self.show_categories()
                elif choice == '5':
                    entry_id = int(input("Enter entry ID: "))
                    self.show_entry_details(entry_id)
                elif choice == '6':
                    search_term = input("Enter search term: ").strip()
                    if search_term:
                        self.search_and_display(search_term)
                elif choice == '7':
                    category = input("Enter category name: ").strip()
                    if category:
                        self.show_category_entries(category)
                elif choice == '8':
                    n = int(input("Enter number of entries: "))
                    view_type = input("Enter 'head' or 'tail': ").strip().lower()
                    if view_type == 'head':
                        self.show_head(n)
                    elif view_type == 'tail':
                        self.show_tail(n)
                    else:
                        print("Invalid view type. Use 'head' or 'tail'.")
                elif choice == '9':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please try again.")
                
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except Exception as e:
                print(f"An error occurred: {e}")
            
            print("\nPress Enter to continue...")
            input()
            print()

def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QA Database Visualizer")
    parser.add_argument("--db", default="qa_database.db", help="Database file path")
    parser.add_argument("--head", type=int, metavar="N", help="Show first N entries")
    parser.add_argument("--tail", type=int, metavar="N", help="Show last N entries")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--categories", action="store_true", help="Show all categories")
    parser.add_argument("--entry", type=int, metavar="ID", help="Show entry details by ID")
    parser.add_argument("--search", type=str, metavar="TERM", help="Search for entries")
    parser.add_argument("--category", type=str, metavar="NAME", help="Show entries in category")
    parser.add_argument("--interactive", action="store_true", help="Run interactive menu")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db):
        print(f"Database file '{args.db}' not found.")
        return
    
    visualizer = QADatabaseVisualizer(args.db)
    
    # Handle command line arguments
    if args.head is not None:
        visualizer.show_head(args.head)
    elif args.tail is not None:
        visualizer.show_tail(args.tail)
    elif args.stats:
        visualizer.show_statistics()
    elif args.categories:
        visualizer.show_categories()
    elif args.entry is not None:
        visualizer.show_entry_details(args.entry)
    elif args.search:
        visualizer.search_and_display(args.search)
    elif args.category:
        visualizer.show_category_entries(args.category)
    elif args.interactive:
        visualizer.interactive_menu()
    else:
        # Default behavior - show interactive menu
        visualizer.interactive_menu()

if __name__ == "__main__":
    main()