import cv2
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = 'dataset'
categories = ['confident', 'nervous']

# 1. Create folders if they don't exist
for category in categories:
    path = os.path.join(DATA_DIR, category)
    if not os.path.exists(path):
        os.makedirs(path)

cap = cv2.VideoCapture(0)

print("="*40)
print(" DATA COLLECTOR TOOL")
print("="*40)
print(" Press 'c' to save CONFIDENT image")
print(" Press 'n' to save NERVOUS image")
print(" Press 'q' to QUIT")
print("="*40)

counts = {cat: len(os.listdir(os.path.join(DATA_DIR, cat))) for cat in categories}

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # Show counts on screen
    cv2.putText(display_frame, f"Confident: {counts['confident']}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Nervous:   {counts['nervous']}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Data Collector', display_frame)

    key = cv2.waitKey(1) & 0xFF

    # SAVE LOGIC
    if key == ord('c'):
        # Save Confident
        filename = os.path.join(DATA_DIR, 'confident', f"{int(time.time())}.jpg")
        cv2.imwrite(filename, frame)
        counts['confident'] += 1
        print(f"[SAVED] Confident Image. Total: {counts['confident']}")
        
    elif key == ord('n'):
        # Save Nervous
        filename = os.path.join(DATA_DIR, 'nervous', f"{int(time.time())}.jpg")
        cv2.imwrite(filename, frame)
        counts['nervous'] += 1
        print(f"[SAVED] Nervous Image.   Total: {counts['nervous']}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()