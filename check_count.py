import os

DATA_DIR = 'dataset'
categories = ['confident', 'nervous']

print(f"\nScanning '{DATA_DIR}' folder...\n")

total_images = 0

if os.path.exists(DATA_DIR):
    for category in categories:
        folder_path = os.path.join(DATA_DIR, category)
        if os.path.exists(folder_path):
            count = len(os.listdir(folder_path))
            print(f"📁 {category.upper()}: Found {count} images")
            total_images += count
        else:
            print(f"❌ {category.upper()}: Folder not found!")
else:
    print(f"❌ Error: The folder '{DATA_DIR}' does not exist here.")

print("-" * 30)
print(f"TOTAL IMAGES: {total_images}")
print("-" * 30)