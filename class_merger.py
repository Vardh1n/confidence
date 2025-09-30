import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def cluster_classes_and_update_db(db_path: str, class_column: str = "classes"):
    """
    Clusters class names into 2, 4, and 8 clusters and adds new columns to the DB.
    Args:
        db_path: Path to the SQLite database.
        class_column: Name of the original class column.
    """
    # Step 1: Fetch all unique classes
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT {class_column} FROM qa_entries")
        class_names = [row[0] for row in cursor.fetchall() if row[0] is not None]

    # Step 2: Embed class names
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(class_names)

    # Step 3: Cluster and assign cluster labels
    cluster_levels = [2, 4, 8]
    cluster_labels = {}
    for k in cluster_levels:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        cluster_labels[k] = {class_name: f"class_{k}_{label}" for class_name, label in zip(class_names, labels)}

    # Step 4: Add new columns to DB if not exist
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for k in cluster_levels:
            col_name = f"class_{k}"
            cursor.execute(f"PRAGMA table_info(qa_entries)")
            columns = [row[1] for row in cursor.fetchall()]
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE qa_entries ADD COLUMN {col_name} TEXT")

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

    print("Database updated with clustered class columns: class_2, class_4, class_8.")


if __name__ == "__main__":
    # Example usage:
    cluster_classes_and_update_db("qa_database.db", class_column="classes")