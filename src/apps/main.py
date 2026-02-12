import sys
import os
import json
import cv2
import time

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.io.video import open_video
from src.core.detector import ProductDetector
from src.core.zone_creator import create_zones
from src.io.storage import save_zones

# CONFIGURATION
MODEL_XML = "models/yolov8x.xml"  # Using the Extra Large Model
LABELS_FILE = "models/labels.json"
OUTPUT_FILE = "data/zones/precise_zones.json"

def main():
    # 1. Load Labels
    with open(LABELS_FILE) as f:
        labels = json.load(f)

    # 2. Initialize (This might take 5-10 seconds for the large model)
    print("⏳ Initializing High-Precision AI... Please wait.")
    cap = open_video(0)
    detector = ProductDetector(MODEL_XML)
    print("✅ AI Ready. High Resolution Mode (1280px) Active.")

    print("\n--- INSTRUCTIONS ---")
    print("1. Point camera at shelf/products.")
    print("2. Press [SPACE] to Scan.")
    print("   (Note: Scanning will take ~1-2 seconds per click)")
    print("3. Press [Q] to Quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Precision Zone Creator", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '): # Spacebar
            print("📸 Scanning... (Holding frame)")
            
            start_time = time.time()
            
            # Run AI (Heavy Computation)
            raw_output = detector.infer(frame)
            zones = create_zones(raw_output, frame.shape, labels)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if zones:
                save_zones(zones, OUTPUT_FILE)
                print(f"✨ Found {len(zones)} items in {duration:.2f}s")
                print(f"📁 Saved to {OUTPUT_FILE}")
            else:
                print("⚠️ No confident detections found.")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()