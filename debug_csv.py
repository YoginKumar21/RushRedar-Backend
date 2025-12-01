import pandas as pd
import os

print("--- DEBUGGING CSV ---")
file_path = 'bangalore_traffic.csv'

if not os.path.exists(file_path):
    print(f"❌ ERROR: File '{file_path}' not found in this folder.")
else:
    try:
        df = pd.read_csv(file_path)
        print("✅ File loaded successfully.")
        print(f"Columns found: {list(df.columns)}")
        
        # Check why the app is rejecting it
        if 'Date' in df.columns and 'Time' in df.columns and 'Total' in df.columns:
            print("✅ Format matches 'Original Schema' (Date, Time, Total).")
        elif 'Traffic_Volume' in df.columns and 'Date' in df.columns:
            print("✅ Format matches 'Unified Schema' (Traffic_Volume, Date).")
        else:
            print("❌ FAILURE: The columns do not match what app.py expects!")
            print("   Expected either: ['Date', 'Time', 'Total']")
            print("   OR: ['Date', 'Traffic_Volume']")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR reading CSV: {e}")