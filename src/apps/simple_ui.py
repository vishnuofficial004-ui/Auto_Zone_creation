import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import sys
import os
import json
import math
import numpy as np # Needed for color checking

# Add 'src' to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.io.video import open_video
from src.core.detector import ProductDetector
from src.core.zone_creator import create_zones
from src.io.storage import save_zones

# --- CONFIGURATION ---
MODEL_PATH = "models/yolov8x.xml"
LABELS_PATH = "models/labels.json"

class RetailAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retail Auto-Zone (Production PoC)")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        # Layout Setup
        self.video_frame = tk.Frame(root, bg="black")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True)

        self.controls = tk.Frame(root, width=350, bg="white")
        self.controls.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls.pack_propagate(False)

        tk.Label(self.controls, text="Control Panel", font=("Segoe UI", 16, "bold"), bg="white").pack(pady=20)
        
        self.btn_detect = tk.Button(self.controls, text="🔍 DETECT ALL", 
                                    font=("Segoe UI", 12, "bold"), bg="#28a745", fg="white",
                                    height=2, command=self.run_full_detection)
        self.btn_detect.pack(fill=tk.X, padx=20, pady=10)

        self.btn_retry = tk.Button(self.controls, text="➕ RETRY / ADD MISSED", 
                                   font=("Segoe UI", 12, "bold"), bg="#007bff", fg="white",
                                   height=2, command=self.run_retry)
        self.btn_retry.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_status = tk.Label(self.controls, text="Initializing...", fg="gray", bg="white", font=("Segoe UI", 10))
        self.lbl_status.pack(pady=10)

        self.result_text = tk.Text(self.controls, height=15, width=35, font=("Consolas", 10), bg="#f8f9fa", relief=tk.FLAT)
        self.result_text.pack(padx=20)

        # Variables
        self.zones = [] 
        self.cap = None
        self.model = None
        self.labels = {}
        self.current_frame = None

        # Load AI
        self.lbl_status.config(text="⏳ Loading AI... (App will freeze for 10s)", fg="orange")
        self.root.update() 
        self.load_ai()

        # Start Camera
        try:
            self.cap = open_video(0)
            self.update_video()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not open camera: {e}")

    def load_ai(self):
        try:
            print("⏳ Loading Model...")
            self.model = ProductDetector(MODEL_PATH)
            with open(LABELS_PATH) as f:
                self.labels = json.load(f)
            self.lbl_status.config(text="✅ AI Ready. System Online.", fg="green")
        except Exception as e:
            self.lbl_status.config(text=f"❌ AI Failed: {e}", fg="red")

    def update_video(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                display_img = frame.copy()
                
                for z in self.zones:
                    x, y, w, h = z['bbox']
                    # Draw existing zones
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_img, z['product'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                if w > 800:
                    scale = 800/w
                    rgb = cv2.resize(rgb, (800, int(h*scale)))
                
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_video) 

    def run_full_detection(self):
        """Wipes everything and starts fresh"""
        if not self.model: return
        self.lbl_status.config(text="📸 Scanning...", fg="blue")
        self.root.update()
        
        raw = self.model.infer(self.current_frame)
        self.zones = create_zones(raw, self.current_frame.shape, self.labels, conf_thresh=0.50)
        
        # --- NEW: Save the Average Color of the detected zone ---
        self.attach_color_signatures(self.zones)
        
        save_zones(self.zones, "data/zones/gui_zones.json")
        self.refresh_summary()
        self.lbl_status.config(text=f"✅ Detected {len(self.zones)} products.", fg="green")

    def run_retry(self):
        """Adds missed items AND removes 'Ghosts' (Moved items)"""
        if not self.model: return
        self.lbl_status.config(text="🔎 Refinement Scan...", fg="blue")
        self.root.update()

        # 1. CLEANUP STEP: Remove zones that changed drastically (Ghost Check)
        valid_zones = []
        removed_count = 0
        
        for z in self.zones:
            if self.is_zone_still_valid(z):
                valid_zones.append(z)
            else:
                removed_count += 1
        
        self.zones = valid_zones # Update list with valid ones only

        # 2. ADD STEP: Find new items
        raw = self.model.infer(self.current_frame)
        candidates = create_zones(raw, self.current_frame.shape, self.labels, conf_thresh=0.35)
        
        added_count = 0
        for cand in candidates:
            if self.is_new_item(cand):
                self.zones.append(cand)
                added_count += 1
        
        # Attach colors to new items
        self.attach_color_signatures(self.zones)
        
        save_zones(self.zones, "data/zones/gui_zones.json")
        self.refresh_summary()
        
        status_msg = ""
        if added_count > 0: status_msg += f"✨ Added {added_count} new. "
        if removed_count > 0: status_msg += f"🗑️ Removed {removed_count} moved. "
        
        if status_msg == "": status_msg = "No changes found."
        self.lbl_status.config(text=status_msg, fg="green")

    def attach_color_signatures(self, zones):
        """Calculates the average color of the zone area and saves it"""
        for z in zones:
            if 'avg_color' in z: continue # Already calculated
            x, y, w, h = z['bbox']
            # Safety crop
            x, y = max(0, x), max(0, y)
            
            roi = self.current_frame[y:y+h, x:x+w]
            if roi.size > 0:
                # Calculate mean color (B, G, R)
                avg_color = np.mean(roi, axis=(0, 1))
                z['avg_color'] = avg_color.tolist()

    def is_zone_still_valid(self, zone):
        """Checks if the zone looks the same as when it was detected"""
        if 'avg_color' not in zone: return True
        
        x, y, w, h = zone['bbox']
        roi = self.current_frame[y:y+h, x:x+w]
        
        if roi.size == 0: return False # Off screen

        current_avg = np.mean(roi, axis=(0, 1))
        saved_avg = np.array(zone['avg_color'])
        
        # Calculate Color Distance (Euclidean)
        dist = np.linalg.norm(current_avg - saved_avg)
        
        # Threshold: If color shifts by > 45 units, assume object moved.
        # (e.g., Black Phone -> Red Curtain is a HUGE shift)
        if dist > 45:
            return False # Delete this zone
        return True

    def is_new_item(self, candidate):
        cx_new = candidate['bbox'][0] + candidate['bbox'][2]/2
        cy_new = candidate['bbox'][1] + candidate['bbox'][3]/2

        for existing in self.zones:
            cx_old = existing['bbox'][0] + existing['bbox'][2]/2
            cy_old = existing['bbox'][1] + existing['bbox'][3]/2
            dist = math.sqrt((cx_new - cx_old)**2 + (cy_new - cy_old)**2)
            if dist < 60: return False
        return True

    def refresh_summary(self):
        self.result_text.delete(1.0, tk.END)
        counts = {}
        for z in self.zones:
            p = z['product']
            counts[p] = counts.get(p, 0) + 1
        
        text = f"TOTAL ZONES: {len(self.zones)}\n" + "-"*20 + "\n"
        for p, c in counts.items(): text += f"{p.upper()}: {c}\n"
        self.result_text.insert(tk.END, text)

if __name__ == "__main__":
    root = tk.Tk()
    app = RetailAIApp(root)
    root.mainloop()