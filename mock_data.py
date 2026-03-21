import sqlite3
import random
from datetime import datetime, timedelta

def inject_mock_data():
    conn = sqlite3.connect('dakshboard.db')
    c = conn.cursor()
    email = "local_athlete"
    
    # Start 45 days ago
    start_date = datetime.now() - timedelta(days=45)
    
    for i in range(45):
        current_date = start_date + timedelta(days=i)
        
        # Simulate a standard marathon training block
        if i % 7 == 0: 
            continue # Rest day (Monday)
            
        if i % 7 == 6: 
            trimp = random.randint(150, 220) # Sunday Long Run
            workout = "Mixed Aerobic Base"
        elif i % 7 == 3: 
            trimp = random.randint(120, 160) # Wednesday Tempo
            workout = "Tempo Run"
        else: 
            trimp = random.randint(50, 90) # Standard Z2 days
            workout = "Recovery / Easy Aerobic"
            
        date_str = current_date.strftime("%Y-%m-%d 08:00")
        filename = f"mock_run_day_{i}.fit"
        
        try:
            c.execute("INSERT INTO user_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (email, filename, date_str, "10.0 km", trimp, workout, "5:30 /km", 2.5))
        except sqlite3.IntegrityError:
            pass
            
    conn.commit()
    conn.close()
    print("✅ 45 Days of Mock Marathon Training Injected!")

if __name__ == "__main__":
    inject_mock_data()