import os
import psycopg2
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
                        val = parts[1].strip().strip('"').strip("'")
                        os.environ[key] = val

# Load .env
load_env()

# Database credentials (defaulting to local fallback values)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Inba@2005")
DB_NAME = os.getenv("DB_NAME", "TamiltoSpeech")
DATABASE_URL = os.getenv("DATABASE_URL")

def create_database_if_not_exists():
    """Connects to the default 'postgres' database and creates DB_NAME if it doesn't exist."""
    if DATABASE_URL:
        # Remote databases are pre-created, bypass this step
        return
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname="postgres"
        )
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if the database exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s;", (DB_NAME,))
        exists = cur.fetchone()
        if not exists:
            print(f"Database '{DB_NAME}' does not exist. Creating it...")
            cur.execute(f'CREATE DATABASE "{DB_NAME}";')
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        cur.close()
    except Exception as e:
        print(f"Error checking/creating database: {e}")
    finally:
        if conn:
            conn.close()

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        cursor_factory=RealDictCursor
    )

def init_db():
    """Initializes the database by creating it first, then creating the users table."""
    # Step 1: Ensure database exists (if not using a connection string)
    if not DATABASE_URL:
        create_database_if_not_exists()
    
    # Step 2: Ensure users table exists
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
        print("Database initialized successfully: 'users' table is ready.")
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
