import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    # 1. Runs tied to an email/ID
    c.execute('''CREATE TABLE IF NOT EXISTS user_runs
                 (email TEXT, filename TEXT, date TEXT, distance TEXT, trimp INTEGER,
                  workout_type TEXT, avg_pace TEXT, drift REAL,
                  UNIQUE(email, filename))''')
                  
    # 2. Athlete Profiles tied to an email/ID (These are the missing functions!)
    c.execute('''CREATE TABLE IF NOT EXISTS user_profiles
                 (email TEXT UNIQUE, age INTEGER, weight REAL, height INTEGER, max_hr INTEGER, rest_hr INTEGER)''')
    conn.commit()
    conn.close()

def get_user_profile(email):
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    c.execute("SELECT age, weight, height, max_hr, rest_hr FROM user_profiles WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    return row

def save_user_profile(email, age, weight, height, max_hr, rest_hr):
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO user_profiles (email, age, weight, height, max_hr, rest_hr)
                 VALUES (?, ?, ?, ?, ?, ?)''', (email, age, weight, height, max_hr, rest_hr))
    conn.commit()
    conn.close()

def run_exists(email, filename):
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    c.execute("SELECT 1 FROM user_runs WHERE email=? AND filename=?", (email, filename))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def save_run(email, filename, date, distance, trimp, workout_type, avg_pace, drift):
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO user_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (email, filename, date, distance, trimp, workout_type, avg_pace, drift))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Ignore duplicates
    conn.close()

def load_history(email):
    conn = sqlite3.connect('dakshboard.db')
    df = pd.read_sql_query("SELECT filename, date, distance, trimp, workout_type, avg_pace, drift FROM user_runs WHERE email=? ORDER BY date DESC", conn, params=(email,))
    conn.close()
    return df