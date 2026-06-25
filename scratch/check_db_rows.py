import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db import get_db_connection

def check_rows():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, fullname, email, created_at FROM users;")
        rows = cur.fetchall()
        print(f"Total rows found in Python (port 5433): {len(rows)}")
        for row in rows:
            print(dict(row))
        cur.close()
    except Exception as e:
        print(f"Error checking rows: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_rows()
