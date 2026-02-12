import json
import os

def save_zones(zones, output_path):
    # Ensure the folder exists (e.g., data/zones)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the file
    with open(output_path, "w") as f:
        json.dump(zones, f, indent=4)
    
    # Print confirmation
    print(f"✅ Saved {len(zones)} zones to: {output_path}")