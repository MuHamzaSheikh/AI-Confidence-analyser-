import os
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'dataset'
OUTPUT_FILE = 'sorted_dataset.xlsx'

print("Scanning folders...")

# 1. Get the list of files for each emotion
conf_path = os.path.join(DATA_DIR, 'confident')
nerv_path = os.path.join(DATA_DIR, 'nervous')

# Safety check: ensure folders exist before reading
if os.path.exists(conf_path):
    conf_list = os.listdir(conf_path)
else:
    conf_list = []

if os.path.exists(nerv_path):
    nerv_list = os.listdir(nerv_path)
else:
    nerv_list = []

# 2. Create the Table
# We use pd.Series because it handles lists of different lengths automatically
df = pd.DataFrame({
    'Confident Images': pd.Series(conf_list),
    'Nervous Images': pd.Series(nerv_list)
})

# 3. Save to Excel
try:
    df.to_excel(OUTPUT_FILE, index=False)
    print("\n" + "="*40)
    print(f" SUCCESS! Created: {OUTPUT_FILE}")
    print("="*40)
    print(f" Confident Count: {len(conf_list)}")
    print(f" Nervous Count:   {len(nerv_list)}")
    print("="*40)
except ImportError:
    print("\n[ERROR] Missing Excel library.")
    print("Please run this command: pip install openpyxl")
except PermissionError:
    print(f"\n[ERROR] Close '{OUTPUT_FILE}' and try again.")