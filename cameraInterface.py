import tkinter as tk  # Add this import at the top with other imports
from tkinter import ttk, messagebox, simpledialog
import tkinter.font as tkfont
from datetime import datetime
import re
import cv2
import numpy as np
import threading
from PIL import Image, ImageTk
import mediapipe as mp
from ultralytics import YOLO
import os
import time
from queue import Queue
import sounddevice as sd
import wave
import subprocess
import geocoder
import pyautogui

def replace_string_in_file(file_path, search_string, replace_string):
    with open(file_path, 'r') as file:
        file_contents = file.read()

    new_contents = file_contents.replace(search_string, replace_string)

    with open(file_path, 'w') as file:
        file.write(new_contents)

# Constants
SAVE_DIR = "vids"
os.makedirs(SAVE_DIR, exist_ok=True)
AUDIO_RATE, CHANNELS = 44100, 1
FPS = 30  # Default FPS value

class PhoneNumberWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Emergency Contacts")
        self.window.geometry("400x500")
        self.window.configure(bg="#f0f2f5")
        
        # Phone numbers list (list of tuples: (name, number))
        self.phone_numbers = []
        
        # Main container
        self.main_frame = tk.Frame(self.window, bg="#f0f2f5")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ttk.Label(self.main_frame,
                                   text="Emergency Contacts",
                                   style="Header.TLabel")
        self.title_label.pack(pady=(0, 20))
        
        # Phone number entry
        self.entry_frame = tk.Frame(self.main_frame, bg="#f0f2f5")
        self.entry_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.phone_entry = ttk.Entry(self.entry_frame,
                                   font=('Helvetica', 12),
                                   width=20)
        self.phone_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        self.add_btn = ttk.Button(self.entry_frame,
                                text="Add Contact",
                                command=self.add_phone_number,
                                style="Contacts.TButton")
        self.add_btn.pack(side=tk.LEFT)
        
        # Phone numbers list
        self.list_frame = tk.Frame(self.main_frame, bg="#ffffff")
        self.list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.phone_list = tk.Listbox(self.list_frame,
                                   font=('Helvetica', 12),
                                   bg="#ffffff",
                                   relief=tk.FLAT,
                                   highlightthickness=1,
                                   highlightbackground="#e0e0e0")
        self.phone_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Delete button
        self.delete_btn = ttk.Button(self.main_frame,
                                   text="Delete Selected",
                                   command=self.delete_phone_number,
                                   style="Stop.TButton")
        self.delete_btn.pack(pady=10)
        
        # Load saved numbers
        self.load_phone_numbers()
    
    def validate_phone_number(self, number):
        # Indian phone number validation (10 digits starting with 6-9)
        pattern = r'^[6-9]\d{9}$'
        return bool(re.match(pattern, number))
    
    def add_phone_number(self):
        number = self.phone_entry.get().strip()
        if not number:
            messagebox.showerror("Error", "Please enter a phone number")
            return
        if not self.validate_phone_number(number):
            messagebox.showerror("Error", "Please enter a valid Indian phone number (10 digits starting with 6-9)")
            return
        # Ask for name
        name = simpledialog.askstring("Contact Name", "Enter the name for this contact:", parent=self.window)
        if not name or not name.strip():
            messagebox.showerror("Error", "Please enter a name for the contact")
            return
        name = name.strip()
        # Check for duplicates
        for n, num in self.phone_numbers:
            if num == number:
                messagebox.showerror("Error", "This number is already in the list")
                return
        self.phone_numbers.append((name, number))
        
        self.phone_list.insert(tk.END, f"{name} - {number}")
        self.phone_entry.delete(0, tk.END)
        self.save_phone_numbers()
    
    def delete_phone_number(self):
        try:
            selection = self.phone_list.curselection()
            if selection:
                index = selection[0]
                self.phone_numbers.pop(index)
                self.phone_list.delete(index)
                self.save_phone_numbers()
        except Exception as e:
            messagebox.showerror("Error", "Please select a number to delete")
    
    def save_phone_numbers(self):
        with open("emergency_contacts.txt", "w") as f:
            for name, number in self.phone_numbers:
                f.write(f"{name}|{number}\n")
    
    def load_phone_numbers(self):
        try:
            with open("emergency_contacts.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        if '|' in line:
                            name, number = line.split('|', 1)
                            if self.validate_phone_number(number):
                                self.phone_numbers.append((name, number))
                                self.phone_list.insert(tk.END, f"{name} - {number}")
        except FileNotFoundError:
            pass

class SOSMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Women Safety Guardian: IndriyaRaksha")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f2f5")
        
        # Custom styles
        self.style = ttk.Style()
        self.style.configure("Start.TButton", padding=15, font=('Helvetica', 11, 'bold'), background="#4CAF50", foreground="black")
        self.style.configure("Stop.TButton", padding=15, font=('Helvetica', 11, 'bold'), background="#f44336", foreground="black")
        self.style.configure("Emergency.TButton", padding=15, font=('Helvetica', 13, 'bold'), background="#ff1744", foreground="black")
        self.style.configure("Contacts.TButton", padding=15, font=('Helvetica', 11, 'bold'), background="#2196F3", foreground="black")
        self.style.configure("Custom.TButton", padding=15, font=('Helvetica', 11, 'bold'), background="#9C27B0", foreground="black")
        self.style.configure("Status.TLabel", font=('Helvetica', 11), background="#ffffff", padding=8)
        self.style.configure("Header.TLabel", font=('Helvetica', 18, 'bold'), background="#f0f2f5", foreground="#1a237e")
        self.style.configure("Parameter.TLabel", font=('Helvetica', 12, 'bold'), background="#ffffff", padding=5)
        self.style.configure("Value.TLabel", font=('Helvetica', 12), background="#ffffff", padding=5)

        # Main content frame using grid
        self.main_content = tk.Frame(self.root, bg="#f0f2f5")
        self.main_content.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Header
        self.header = tk.Frame(self.main_content, bg="#f0f2f5")
        self.header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(10, 10))
        self.header.grid_columnconfigure(0, weight=1)
        self.header.grid_columnconfigure(1, weight=1)
        self.title_label = ttk.Label(self.header, text="Women Safety Guardian : INDRIYA RAKSHA", style="Header.TLabel")
        self.title_label.grid(row=0, column=0, sticky="w", padx=(30, 0))
        self.time_label = ttk.Label(self.header, text="", style="Status.TLabel")
        self.time_label.grid(row=0, column=1, sticky="e", padx=(0, 30))
        self.update_time()

        # Status Card (centered)
        self.status_card = tk.Frame(self.main_content, bg="#ffffff", relief=tk.FLAT, highlightthickness=1, highlightbackground="#e0e0e0")
        self.status_card.grid(row=1, column=0, sticky="nsew", padx=(30, 15), pady=(0, 15))
        self.main_content.grid_rowconfigure(1, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)

        # Make the parameters section scrollable if needed
        self.param_canvas = tk.Canvas(self.status_card, bg="#ffffff", highlightthickness=0, height=350)
        self.param_scrollbar = ttk.Scrollbar(self.status_card, orient="vertical", command=self.param_canvas.yview)
        self.param_scrollable_frame = tk.Frame(self.param_canvas, bg="#ffffff")
        self.param_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))
        )
        self.param_canvas.create_window((0, 0), window=self.param_scrollable_frame, anchor="nw")
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        self.param_canvas.pack(side="left", fill="both", expand=True)
        self.param_scrollbar.pack(side="right", fill="y")

        self.param_scrollable_frame.grid_columnconfigure(0, weight=1)
        self.param_scrollable_frame.grid_columnconfigure(1, weight=1)
        # Use the provided status_fields for initialization
        status_fields = [
            ("Emotion", "N/A"),
            ("Confidence", "0%"),
            ("Decibel Level", "0 dB"),
            ("SOS Detected", "No"),
            ("Location", "Acquiring..."),
            ("Crime Zone", "Checking..."),
            ("Video Model Triggered", "No"),
            ("Sensor", "Active"),
            ("Monitoring", "Inactive")
        ]
        self.labels = {}
        for i, (field, value) in enumerate(status_fields):
            label = ttk.Label(self.param_scrollable_frame, text=f"{field}:", style="Parameter.TLabel", anchor="center", justify="center")
            label.grid(row=i, column=0, sticky="e", padx=(0, 10), pady=10)
            value_label = ttk.Label(self.param_scrollable_frame, text=value, style="Value.TLabel", anchor="center", justify="center")
            value_label.grid(row=i, column=1, sticky="w", pady=10)
            self.labels[field] = value_label

        # Camera Feed Section (centered, visually prominent)
        self.setup_camera_frame()
        
        # Log Section (centered)
        self.log_frame = tk.Frame(self.main_content, bg="#ffffff")
        self.log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=(30, 15), pady=(0, 15))
        self.main_content.grid_rowconfigure(2, weight=1)
        self.log_header = ttk.Label(self.log_frame, text="System Logs", style="Header.TLabel")
        self.log_header.pack(anchor="w", padx=15, pady=10)
        self.log_container = tk.Frame(self.log_frame, bg="#ffffff")
        self.log_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.log_box = tk.Text(self.log_container, height=10, wrap=tk.WORD, font=('Consolas', 11), bg="#ffffff", relief=tk.FLAT, highlightthickness=1, highlightbackground="#e0e0e0")
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar = ttk.Scrollbar(self.log_container, orient="vertical", command=self.log_box.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_box.configure(yscrollcommand=self.log_scrollbar.set)
        self.log_box.insert(tk.END, "System Ready...\n")

        # Fixed Button Section at the bottom
        self.button_frame = tk.Frame(self.root, bg="#f0f2f5", height=80)
        self.button_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        self.root.grid_rowconfigure(1, weight=0)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.left_buttons = tk.Frame(self.button_frame, bg="#f0f2f5")
        self.left_buttons.grid(row=0, column=0, sticky="w")
        self.start_btn = ttk.Button(self.left_buttons, text="▶ Start Monitoring", command=self.start_monitoring, style="Start.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(self.left_buttons, text="■ Stop Monitoring", command=self.stop_monitoring, style="Stop.TButton", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.right_buttons = tk.Frame(self.button_frame, bg="#f0f2f5")
        self.right_buttons.grid(row=0, column=1, sticky="e")
        self.contacts_btn = ttk.Button(self.right_buttons, text="📱 Emergency Contacts", command=self.open_phone_window, style="Contacts.TButton")
        self.contacts_btn.pack(side=tk.LEFT, padx=5)
        self.custom_word_btn = ttk.Button(self.right_buttons, text="🔔 Custom Alert Word", command=self.set_custom_word, style="Custom.TButton")
        self.custom_word_btn.pack(side=tk.LEFT, padx=5)
        self.emergency_btn = ttk.Button(self.right_buttons, text="🚨 EMERGENCY SOS", command=self.trigger_emergency, style="Emergency.TButton")
        self.emergency_btn.pack(side=tk.LEFT, padx=5)

        # Replace the existing camera frame setup with our new method
        self.setup_camera_frame()

    def open_phone_window(self):
        PhoneNumberWindow(self.root)

    def set_custom_word(self):
        custom_word = tk.simpledialog.askstring("Custom Alert Word",
                                              "Enter a custom word to trigger alerts:",
                                              parent=self.root)
        if custom_word:
            self.log(f"Custom alert word set to: {custom_word}")
            # Add your custom word handling logic here

    def update_time(self):
        current_time = datetime.now().strftime("%I:%M:%S %p")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def start_monitoring(self):
        # Create camera attributes
        self.camera_running = True
        self.is_recording = False
        self.frames = []
        self.model = YOLO("yolov8n.pt")
        self.last_sos = 0
        self.cooldown = 3
        self.fps = 30
        
        # Create the camera thread
        self.camera_thread = threading.Thread(target=self.update_camera_feed, daemon=True)
        self.camera_thread.start()
        
        self.update_status("Monitoring", "Active")
        self.log("Monitoring started.")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_monitoring(self):
        self.camera_running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1.0)
        
        # Clear the camera views
        for label in self.view_labels.values():
            label.configure(image='')
            
        self.update_status("Monitoring", "Stopped")
        self.log("Monitoring stopped.")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def setup_camera_frame(self):
        # Modify the camera frame to hold 4 views
        self.camera_frame = tk.Frame(self.main_content, bg="#222")
        self.camera_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 5), pady=(0, 5))
        self.main_content.grid_columnconfigure(1, weight=4)  # Give even more weight to camera column
        
        # Configure grid for camera frame - remove spacing between cells
        self.camera_frame.grid_columnconfigure(0, weight=1, uniform="col")
        self.camera_frame.grid_columnconfigure(1, weight=1, uniform="col")
        self.camera_frame.grid_rowconfigure(0, weight=1, uniform="row")
        self.camera_frame.grid_rowconfigure(1, weight=1, uniform="row")
        
        # Create labels for each view with titles
        view_titles = {
            "Color": "Original Feed",
            "Gray": "Grayscale View",
            "Edges": "Edge Detection",
            "Infra": "Infrared View"
        }
        
        self.view_labels = {}
        row, col = 0, 0
        for name, title in view_titles.items():
            frame = tk.Frame(self.camera_frame, bg="#222")
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(0, minsize=20)  # Fixed height for title
            frame.grid_rowconfigure(1, weight=1)
            
            # Title label
            title_label = ttk.Label(frame, text=title, foreground="#fff", background="#222",
                                  font=("Helvetica", 10, "bold"))
            title_label.grid(row=0, column=0, pady=(5,2))
            
            # Camera view label
            self.view_labels[name] = ttk.Label(frame)
            self.view_labels[name].grid(row=1, column=0)
            
            col += 1
            if col > 1:
                col = 0
                row += 1

    def update_camera_feed(self):
        # Initialize MediaPipe hands
        mp_hands = mp.solutions.hands
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        missing = []  # We'll use empty list as default
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Larger resolution
        cap.set(4, 480)  # Larger resolution
        self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.7) as hands:
            while self.camera_running:
                ret, img = cap.read()
                if not ret:
                    continue
                    
                img = cv2.flip(img, 1)
                ori = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 70, 200)
                
                views = {
                    "Color": ori.copy(),
                    "Gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                    "Edges": cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                    "Infra": np.zeros_like(img)
                }
                views["Infra"][:, :, 1] = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                
                # Check for SOS conditions
                sos = False
                
                # Hand detection
                res = hands.process(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    # Check for fist gesture
                    lm = res.multi_hand_landmarks[0].landmark
                    folded = [f for f in fingers if f not in missing and 
                             lm[{"thumb":4,"index":8,"middle":12,"ring":16,"pinky":20}[f]].y > 
                             lm[{"thumb":3,"index":6,"middle":10,"ring":14,"pinky":18}[f]].y]
                    if len(folded) == (5 - len(missing)):
                        sos = True
                
                # Object detection
                yres = self.model.predict(ori, verbose=False)[0]
                for b in yres.boxes:
                    if self.model.names[int(b.cls[0])] in ["knife", "scissors"] and float(b.conf[0]) > 0.4:
                        sos = True
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cv2.rectangle(views["Color"], (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Handle SOS detection
                if sos and time.time() - self.last_sos > self.cooldown:
                    self.last_sos = time.time()
                    self.trigger_emergency()
                
                # Handle recording
                if self.is_recording:
                    self.frames.append(ori.copy())
                    if time.time() - self.recording_start >= 5:
                        self.st.set()
                        time.sleep(0.2)
                        audio = self.q.get()
                        merge_save(self.frames, audio)
                        self.is_recording = False
                        self.frames = []
                
                # Update the view labels
                for name, image in views.items():
                    if sos:
                        cv2.putText(image, "!!! SOS !!!", (10, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Convert to RGB for tkinter
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (480, 360))  # Even larger view size
                    photo = ImageTk.PhotoImage(image=Image.fromarray(image))
                    self.view_labels[name].configure(image=photo)
                    self.view_labels[name].image = photo
        
        cap.release()

    def trigger_emergency(self):
        self.log("🚨 EMERGENCY SOS TRIGGERED!")
        self.update_status("SOS Detected", "YES")
        
        # Start recording
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.q = Queue()
            self.st = threading.Event()
            threading.Thread(target=record_audio, args=(self.q, self.st), daemon=True).start()
            self.recording_start = time.time()
        
        # Get location
        g = geocoder.ip("me")
        if g.ok:
            lat, lon = g.latlng
            location = f"https://maps.google.com/?q={lat},{lon}"
            self.update_status("Location", location)
            msg = f"🚨SOS! My location: {location}"
        else:
            msg = "🚨SOS! Location unavailable."
            self.update_status("Location", "Unavailable")

        # Load emergency contacts
        try:
            with open("emergency_contacts.txt", "r") as f:
                contacts = [line.strip().split("|") for line in f.readlines()]
            
            if contacts:
                self.log("📱 Alerting emergency contacts...")
                for name, number in contacts:
                    self.log(f"Sending alert to {name}")
                    try:
                        thread = threading.Thread(
                            target=lambda: WhatsAppAutomation.send_message(name, msg, disable_failsafe=True),
                            daemon=True
                        )
                        thread.start()
                        self.log(f"Alert sent to {name}")
                    except Exception as e:
                        self.log(f"Failed to send alert to {name}: {str(e)}")
        except FileNotFoundError:
            self.log("⚠ No emergency contacts found!")

        # Play alert sound
        threading.Thread(target=lambda: messagebox.showwarning("EMERGENCY", "🚨 SOS Alert Triggered!"), daemon=True).start()

    def update_status(self, field, value):
        """Update status field in the GUI"""
        if field in self.labels:
            self.labels[field].config(text=value)

    def log(self, message):
        """Add a message to the log box"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_box.see(tk.END)

class WhatsAppAutomation:
    # Add these new helper methods at the start of the class
    @staticmethod
    def get_screen_size():
        """Get screen size to help with coordinate calculations"""
        return pyautogui.size()

    @staticmethod
    def find_whatsapp_elements():
        """Try to find WhatsApp elements on screen"""
        screen_width, screen_height = WhatsAppAutomation.get_screen_size()
        
        # Adjust these coordinates based on your screen resolution
        elements = {
            'search_box': (int(screen_width * 0.2), int(screen_height * 0.1)),
            'first_contact': (int(screen_width * 0.2), int(screen_height * 0.2)),
            'message_box': (int(screen_width * 0.5), int(screen_height * 0.9))
        }
        return elements

    @staticmethod
    def open_chat(friend_name, disable_failsafe=False):
        if disable_failsafe:
            pyautogui.FAILSAFE = False
        
        try:
            # Try opening WhatsApp Desktop
            os.system("start whatsapp:")
            time.sleep(5)  # Wait for WhatsApp to open
            
            # Move to and click the search box
            pyautogui.moveTo(200, 100)
            pyautogui.click()
            time.sleep(1)
            
            # Type the contact name
            pyautogui.write(friend_name)
            time.sleep(2)  # Wait longer for search results
            
            # Move to and click the first contact (adjust Y coordinate as needed)
            pyautogui.moveTo(200, 220)  # Move to first contact in list
            time.sleep(0.5)
            pyautogui.click()
            time.sleep(2)  # Wait for chat to open
            
            # Move to message box area (adjust coordinates as needed)
            pyautogui.moveTo(400, 700)  # Move to message input area
            pyautogui.click()
            time.sleep(1)
            
        except Exception as e:
            print(f"Error opening WhatsApp chat: {e}")
            raise e

    @staticmethod
    def send_message(friend_name, message, disable_failsafe=False):
        try:
            WhatsAppAutomation.open_chat(friend_name, disable_failsafe)
            
            # Ensure we're in the message input area
            pyautogui.moveTo(400, 700)  # Move to message input area
            pyautogui.click()
            time.sleep(1)
            
            # Clear any existing text
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.press('delete')
            time.sleep(0.5)
            
            # Type and send the message
            pyautogui.write(message)
            time.sleep(1)
            pyautogui.press('enter')
            time.sleep(2)  # Wait longer for message to send
            
            print(f"[📨] Message sent to {friend_name}!")
            
            # Close WhatsApp
            WhatsAppAutomation.close_whatsapp()
            
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")
            # Try alternative messaging method if available

    @staticmethod
    def close_whatsapp():
        pyautogui.hotkey("alt", "f4")

def record_audio(q, st):
    frames = []
    def cb(i, f, ti, s):
        frames.append(i.copy())
        if st.is_set(): raise sd.CallbackStop()
    with sd.InputStream(samplerate=AUDIO_RATE, channels=CHANNELS, dtype='int16', callback=cb):
        while not st.is_set():
            time.sleep(0.1)
    q.put(np.concatenate(frames))

def merge_save(frames, audio):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp = os.path.join(SAVE_DIR, f"temp_{now}.mp4")
    out = os.path.join(SAVE_DIR, f"sos_{now}.mp4")
    h, w = frames[0].shape[:2]
    v = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    for f in frames:
        v.write(f)
    v.release()

    wav = tmp.replace(".mp4", "_a.wav")
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(audio.tobytes())
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", wav, "-c:v", "copy", "-c:a", "aac", out])
    os.remove(tmp); os.remove(wav)
    print("[🎉] Saved:", out)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SOSMonitorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
