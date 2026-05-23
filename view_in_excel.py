import os
import pandas as pd

# Define paths
DATA_DIR = 'dataset'
CATEGORIES = ['confident', 'nervous']

data = []

print("Scanning folders...")

# Loop through both folders
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    if os.path.exists(path):
        for img_name in os.listdir(path):
            # Add to list
            data.append({
                'Image_Name': img_name,
                'Label': category,
                'Status': 'Ready for Training'
            })

# Create a DataFrame (Table)
df = pd.DataFrame(data)

# Save to Excel
output_file = 'my_dataset_report.xlsx'
df.to_excel(output_file, index=False)

print(f"Success! Created '{output_file}'.")
print(f"Total Images Found: {len(df)}")