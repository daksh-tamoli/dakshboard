import sqlite3
import pandas as pd

def init_db():
    """Creates the database and the training log table if it doesn't exist."""
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    # We use 'filename' as a UNIQUE constraint so we don't save duplicate runs
    c.execute('''CREATE TABLE IF NOT EXISTS runs
                 (filename TEXT UNIQUE, date TEXT, distance TEXT, trimp INTEGER,
                  workout_type TEXT, avg_pace TEXT, drift REAL)''')
    conn.commit()
    conn.close()

def run_exists(filename):
    """Checks if a run has already been processed to save computing power."""
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    c.execute("SELECT 1 FROM runs WHERE filename=?", (filename,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def save_run(filename, date, distance, trimp, workout_type, avg_pace, drift):
    """Saves the top-level metrics of a run into the database."""
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (filename, date, distance, trimp, workout_type, avg_pace, drift))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Silently ignore if the run is already in the database
    conn.close()

def load_history():
    """Pulls the entire training history for the Race Predictor and UI."""
    conn = sqlite3.connect('dakshboard.db')
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY date DESC", conn)
    conn.close()
    return df