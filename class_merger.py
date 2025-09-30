import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import re

def extract_keywords(text):
    """Extract meaningful keywords from text."""
    # Remove common stop words and extract meaningful words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom'}
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    return meaningful_words

def generate_cluster_name(class_names_in_cluster, model):
    """Generate a meaningful name for a cluster based on the class names in it."""
    # Extract all keywords from class names in this cluster
    all_keywords = []
    for class_name in class_names_in_cluster:
        keywords = extract_keywords(class_name)
        all_keywords.extend(keywords)
    
    if not all_keywords:
        return "miscellaneous"
    
    # Count frequency of keywords
    keyword_counts = Counter(all_keywords)
    
    # Get most common meaningful keywords
    common_keywords = [word for word, count in keyword_counts.most_common(5)]
    
    # Try to find the most representative keyword
    if common_keywords:
        # Use the most frequent keyword as base
        base_keyword = common_keywords[0]
        
        # Some mapping to more general terms
        category_mappings = {
            'science': ['physics', 'chemistry', 'biology', 'research', 'scientific', 'experiment'],
            'technology': ['computer', 'programming', 'software', 'tech', 'digital', 'algorithm', 'code'],
            'health': ['medical', 'doctor', 'medicine', 'health', 'disease', 'treatment', 'hospital'],
            'education': ['school', 'student', 'learn', 'study', 'teaching', 'academic', 'university'],
            'sports': ['football', 'basketball', 'soccer', 'game', 'sport', 'player', 'team'],
            'business': ['company', 'work', 'job', 'office', 'business', 'corporate', 'management'],
            'entertainment': ['movie', 'music', 'show', 'entertainment', 'film', 'television', 'media'],
            'food': ['cooking', 'recipe', 'food', 'restaurant', 'meal', 'kitchen', 'ingredient'],
            'travel': ['travel', 'trip', 'vacation', 'tourism', 'journey', 'destination', 'hotel'],
            'finance': ['money', 'bank', 'investment', 'financial', 'economy', 'budget', 'payment']
        }
        
        # Check if any keywords match our category mappings
        for category, keywords in category_mappings.items():
            if any(keyword in common_keywords for keyword in keywords):
                return category
        
        # If no mapping found, use the most common keyword
        return base_keyword
    
    return "general"

def cluster_category_and_update_db(db_path: str, class_column: str = "category"):
    """
    Clusters class names into 2, 4, and 8 clusters and adds new columns to the DB with meaningful names.
    Args:
        db_path: Path to the SQLite database.
        class_column: Name of the original class column.
    """
    # Step 1: Fetch all unique categories
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT {class_column} FROM qa_entries")
        class_names = [row[0] for row in cursor.fetchall() if row[0] is not None]

    print(f"Found {len(class_names)} unique classes: {class_names[:10]}...")

    # Step 2: Embed class names
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(class_names)

    # Step 3: Cluster and assign cluster labels with meaningful names
    cluster_levels = [2, 4, 8]
    cluster_labels = {}
    
    for k in cluster_levels:
        print(f"\nClustering into {k} groups...")
        kmeans = KMeans(n_clusters=min(k, len(class_names)), random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        
        # Group class names by cluster
        clusters = {}
        for class_name, label in zip(class_names, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(class_name)
        
        # Generate meaningful names for each cluster
        cluster_names = {}
        used_names = set()
        
        for cluster_id, class_names_in_cluster in clusters.items():
            base_name = generate_cluster_name(class_names_in_cluster, model)
            
            # Ensure unique names
            final_name = base_name
            counter = 1
            while final_name in used_names:
                final_name = f"{base_name}_{counter}"
                counter += 1
            
            used_names.add(final_name)
            cluster_names[cluster_id] = final_name
            print(f"  Cluster {cluster_id} -> '{final_name}': {class_names_in_cluster}")
        
        # Create mapping from class names to cluster names
        cluster_labels[k] = {}
        for class_name, label in zip(class_names, labels):
            cluster_labels[k][class_name] = cluster_names[label]

    # Step 4: Add new columns to DB if not exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for k in cluster_levels:
            col_name = f"class_{k}"
            cursor.execute(f"PRAGMA table_info(qa_entries)")
            columns = [row[1] for row in cursor.fetchall()]
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE qa_entries ADD COLUMN {col_name} TEXT")
                print(f"Added column: {col_name}")

        # Step 5: Update each entry with new cluster columns
        for k in cluster_levels:
            col_name = f"class_{k}"
            for class_name in class_names:
                cluster_val = cluster_labels[k][class_name]
                cursor.execute(
                    f"UPDATE qa_entries SET {col_name} = ? WHERE {class_column} = ?",
                    (cluster_val, class_name)
                )
        conn.commit()

    # Display final results
    print(f"\nDatabase updated with clustered class columns:")
    for k in cluster_levels:
        unique_clusters = set(cluster_labels[k].values())
        print(f"  class_{k}: {len(unique_clusters)} clusters -> {sorted(unique_clusters)}")


if __name__ == "__main__":
    # Example usage:
    cluster_category_and_update_db("qa_database.db", class_column="category")