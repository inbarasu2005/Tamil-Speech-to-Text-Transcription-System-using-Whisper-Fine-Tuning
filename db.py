import os
import psycopg2
import sqlite3
from psycopg2.extras import RealDictCursor

# Function to load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        # Do not override existing environment variables (e.g. from Render)
                        if key not in os.environ:
                            val = parts[1].strip().strip('"').strip("'")
                            os.environ[key] = val

# Load .env
load_env()

DATABASE_URL = os.getenv("DATABASE_URL")

class SQLiteCursorWrapper:
    def __init__(self, cursor):
        self.cursor = cursor

    def execute(self, query, params=None):
        # Convert SERIAL PRIMARY KEY to SQLite format
        if "SERIAL PRIMARY KEY" in query:
            query = query.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        
        # Convert %s placeholder to ? for SQLite compatibility
        if params is not None:
            query = query.replace("%s", "?")
            return self.cursor.execute(query, params)
        else:
            return self.cursor.execute(query)

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is not None:
            return dict(row)
        return None

    def fetchall(self):
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self):
        self.cursor.close()

class SQLiteConnectionWrapper:
    def __init__(self, conn):
        self.conn = conn

    def cursor(self):
        return SQLiteCursorWrapper(self.conn.cursor())

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()

def get_db_connection():
    """Establishes a connection to PostgreSQL if DATABASE_URL is set, otherwise falls back to SQLite."""
    if DATABASE_URL:
        # Use PostgreSQL database with a 5-second timeout to prevent hanging the startup phase
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=5)
    else:
        # Fallback to local SQLite database file
        db_path = os.path.join(os.path.dirname(__file__), "local_app.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return SQLiteConnectionWrapper(conn)

def init_db():
    """Initializes the database by creating the users table."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create users table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            fullname VARCHAR(100),
            email VARCHAR(100) UNIQUE,
            password VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        if DATABASE_URL:
            print("Database initialized successfully: PostgreSQL 'users' table is ready.")
        else:
            print("Database initialized successfully: SQLite fallback 'users' table is ready.")
        return True, "Success"
    except Exception as e:
        error_msg = f"Database Initialization Error: {str(e)}"
        print(error_msg)
        return False, error_msg
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Test connection and initialize table
    success, msg = init_db()
    if success:
        print("Test Connection & Setup: SUCCESS")
    else:
        print(f"Test Connection & Setup: FAILED - {msg}")
