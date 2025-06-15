import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime
import re, cv2, numpy as np, threading, os, time, wave, subprocess, geocoder, pyautogui, winsound
from PIL import Image, ImageTk
from ultralytics import YOLO
from queue import Queue
import sounddevice as sd
import mediapipe as mp
import threading
import sounddevice as sd
import numpy as np
from emotion_detection import predict_emotion  # your LSTM model logic
from dbmeter import calculate_decibel


SAVE_DIR, AUDIO_RATE, CHANNELS, FPS = "vids", 44100, 1, 30
os.makedirs(SAVE_DIR, exist_ok=True)

def record_audio(q, st):
    frames = []
    def cb(i, f, ti, s): frames.append(i.copy()); st.is_set() and (_ for _ in ()).throw(sd.CallbackStop())
    with sd.InputStream(samplerate=AUDIO_RATE, channels=CHANNELS, dtype='int16', callback=cb):
        while not st.is_set(): time.sleep(0.1)
    q.put(np.concatenate(frames))

def merge_save(frames, audio):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp, out = os.path.join(SAVE_DIR, f"temp_{now}.mp4"), os.path.join(SAVE_DIR, f"sos_{now}.mp4")
    h, w = frames[0].shape[:2]
    v = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    [v.write(f) for f in frames]; v.release()
    wav = tmp.replace(".mp4", "_a.wav")
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(2); wf.setframerate(AUDIO_RATE); wf.writeframes(audio.tobytes())
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", wav, "-c:v", "copy", "-c:a", "aac", out])
    os.remove(tmp); os.remove(wav)

class PhoneNumberWindow:
    MAX_CONTACTS = 3

    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Emergency Contacts")
        self.window.geometry("400x500")
        self.window.configure(bg="#f0f2f5")
        self.phone_numbers = []
        self.main_frame = tk.Frame(self.window, bg="#f0f2f5")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        ttk.Label(self.main_frame, text="Emergency Contacts (Max 3)", style="Header.TLabel").pack(pady=(0, 20))
        ef = tk.Frame(self.main_frame, bg="#f0f2f5")
        ef.pack(fill=tk.X, pady=(0, 10))
        self.phone_entry = ttk.Entry(ef, font=('Helvetica', 12), width=20)
        self.phone_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(ef, text="Add Contact", command=self.add_phone_number, style="Contacts.TButton").pack(side=tk.LEFT)
        self.list_frame = tk.Frame(self.main_frame, bg="#fff")
        self.list_frame.pack(fill=tk.BOTH, expand=True)
        self.phone_list = tk.Listbox(self.list_frame, font=('Helvetica', 12), bg="#fff", relief=tk.FLAT, highlightthickness=1, highlightbackground="#e0e0e0")
        self.phone_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Button(self.main_frame, text="Delete Selected", command=self.delete_phone_number, style="Stop.TButton").pack(pady=10)
        self.load_phone_numbers()

    def validate_phone_number(self, number):
        return bool(re.match(r'^[6-9]\d{9}$', number))

    def add_phone_number(self):
        if len(self.phone_numbers) >= self.MAX_CONTACTS:
            messagebox.showerror("Error", f"Only {self.MAX_CONTACTS} contacts allowed.")
            return
        number = self.phone_entry.get().strip()
        if not number or not self.validate_phone_number(number):
            return messagebox.showerror("Error", "Please enter a valid Indian phone number (10 digits starting with 6-9)")
        name = simpledialog.askstring("Contact Name", "Enter the name for this contact:", parent=self.window)
        if not name or not name.strip():
            return messagebox.showerror("Error", "Please enter a name for the contact")
        if any(num == number for _, num in self.phone_numbers):
            return messagebox.showerror("Error", "This number is already in the list")
        self.phone_numbers.append((name.strip(), number))
        self.refresh_listbox()
        self.phone_entry.delete(0, tk.END)
        self.save_phone_numbers()

    def delete_phone_number(self):
        try:
            idx = self.phone_list.curselection()[0]
            self.phone_numbers.pop(idx)
            self.refresh_listbox()
            self.save_phone_numbers()
        except:
            messagebox.showerror("Error", "Please select a number to delete")

    def save_phone_numbers(self):
        with open("emergency_contacts.txt", "w") as f:
            for n, num in self.phone_numbers:
                f.write(f"{n}|{num}\n")

    def load_phone_numbers(self):
        try:
            with open("emergency_contacts.txt", "r") as f:
                for line in f:
                    name, number = line.strip().split('|', 1)
                    if self.validate_phone_number(number) and len(self.phone_numbers) < self.MAX_CONTACTS:
                        self.phone_numbers.append((name, number))
            self.refresh_listbox()
        except FileNotFoundError:
            pass

    def refresh_listbox(self):
        self.phone_list.delete(0, tk.END)
        for name, number in self.phone_numbers:
            self.phone_list.insert(tk.END, f"{name} - {number}")

class WhatsAppAutomation:
    @staticmethod
    def open_chat(friend_name, disable_failsafe=False):
        if disable_failsafe: pyautogui.FAILSAFE = False
        os.system("start whatsapp:"); time.sleep(5)
        for x, y in [(200, 100), (200, 200), (700, 700)]:
            pyautogui.moveTo(x, y); pyautogui.click(); time.sleep(1)
            if (x, y) == (200, 100): pyautogui.write(friend_name); time.sleep(2)
    @staticmethod
    def send_message(friend_name, message, disable_failsafe=False):
        try:
            WhatsAppAutomation.open_chat(friend_name, disable_failsafe)
            pyautogui.click(); time.sleep(1); pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete'); time.sleep(0.5)
            pyautogui.write(message); time.sleep(1); pyautogui.press('enter'); time.sleep(2)
            WhatsAppAutomation.close_whatsapp()
        except Exception as e: print(f"Error sending WhatsApp message: {e}")
    @staticmethod
    def close_whatsapp(): pyautogui.hotkey("alt", "f4")

class SOSMonitorApp:
    def __init__(self, root):
        self.root = root; root.title("Women Safety Guardian: IndriyaRaksha"); root.geometry("1000x800"); root.configure(bg="#f0f2f5")
        style = ttk.Style()
        for s, c in [("Start", "#4CAF50"), ("Stop", "#f44336"), ("Emergency", "#ff1744"), ("Contacts", "#2196F3"), ("Custom", "#9C27B0")]:
            style.configure(f"{s}.TButton", padding=15, font=('Helvetica', 11, 'bold'), background=c, foreground="black")
        style.configure("Header.TLabel", font=('Helvetica', 18, 'bold'), background="#f0f2f5", foreground="#1a237e")
        style.configure("Status.TLabel", font=('Helvetica', 11), background="#fff", padding=8)
        style.configure("Parameter.TLabel", font=('Helvetica', 12, 'bold'), background="#fff", padding=5)
        style.configure("Value.TLabel", font=('Helvetica', 12), background="#fff", padding=5)
        main = tk.Frame(root, bg="#f0f2f5"); main.grid(row=0, column=0, sticky="nsew"); root.grid_rowconfigure(0, weight=1); root.grid_columnconfigure(0, weight=1)
        header = tk.Frame(main, bg="#f0f2f5"); header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(10, 10))
        header.grid_columnconfigure(0, weight=1); header.grid_columnconfigure(1, weight=1)
        ttk.Label(header, text="Women Safety Guardian : INDRIYA RAKSHA", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=(30, 0))
        self.time_label = ttk.Label(header, text="", style="Status.TLabel"); self.time_label.grid(row=0, column=1, sticky="e", padx=(0, 30)); self.update_time()
        self.status_card = tk.Frame(main, bg="#fff", relief=tk.FLAT, highlightthickness=1, highlightbackground="#e0e0e0")
        self.status_card.grid(row=1, column=0, sticky="nsew", padx=(30, 15), pady=(0, 15)); main.grid_rowconfigure(1, weight=1); main.grid_columnconfigure(0, weight=1)
        self.param_canvas = tk.Canvas(self.status_card, bg="#fff", highlightthickness=0, height=350)
        self.param_scrollbar = ttk.Scrollbar(self.status_card, orient="vertical", command=self.param_canvas.yview)
        self.param_scrollable_frame = tk.Frame(self.param_canvas, bg="#fff")
        self.param_scrollable_frame.bind("<Configure>", lambda e: self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all")))
        self.param_canvas.create_window((0, 0), window=self.param_scrollable_frame, anchor="nw")
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)
        self.param_canvas.pack(side="left", fill="both", expand=True); self.param_scrollbar.pack(side="right", fill="y")
        self.param_scrollable_frame.grid_columnconfigure(0, weight=1); self.param_scrollable_frame.grid_columnconfigure(1, weight=1)
        status_fields = [("Emotion", "N/A"), ("Confidence", "0%"), ("Decibel Level", "0 dB"), ("SOS Detected", "No"), ("Location", "Acquiring..."), ("Crime Zone", "Checking..."), ("Video Model Triggered", "No"), ("Sensor", "Active"), ("Monitoring", "Inactive")]
        self.labels = {f: ttk.Label(self.param_scrollable_frame, text=v, style="Value.TLabel", anchor="center", justify="center") for i, (f, v) in enumerate(status_fields)}
        [ttk.Label(self.param_scrollable_frame, text=f"{f}:", style="Parameter.TLabel", anchor="center", justify="center").grid(row=i, column=0, sticky="e", padx=(0, 10), pady=10) or self.labels[f].grid(row=i, column=1, sticky="w", pady=10) for i, (f, v) in enumerate(status_fields)]
        self.setup_camera_frame(main)
        self.log_frame = tk.Frame(main, bg="#fff"); self.log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=(30, 15), pady=(0, 15)); main.grid_rowconfigure(2, weight=1)
        ttk.Label(self.log_frame, text="System Logs", style="Header.TLabel").pack(anchor="w", padx=15, pady=10)
        self.log_box = tk.Text(tk.Frame(self.log_frame, bg="#fff").pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15)) or self.log_frame, height=10, wrap=tk.WORD, font=('Consolas', 11), bg="#fff", relief=tk.FLAT, highlightthickness=1, highlightbackground="#e0e0e0")
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); self.log_scrollbar = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_box.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.log_box.configure(yscrollcommand=self.log_scrollbar.set); self.log_box.insert(tk.END, "System Ready...\n")
        self.button_frame = tk.Frame(root, bg="#f0f2f5", height=80); self.button_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10); root.grid_rowconfigure(1, weight=0)
        self.button_frame.grid_columnconfigure(0, weight=1); self.button_frame.grid_columnconfigure(1, weight=1)
        self.left_buttons = tk.Frame(self.button_frame, bg="#f0f2f5"); self.left_buttons.grid(row=0, column=0, sticky="w")
        self.start_btn = ttk.Button(self.left_buttons, text="▶ Start Monitoring", command=self.start_monitoring, style="Start.TButton"); self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(self.left_buttons, text="■ Stop Monitoring", command=self.stop_monitoring, style="Stop.TButton", state=tk.DISABLED); self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.right_buttons = tk.Frame(self.button_frame, bg="#f0f2f5"); self.right_buttons.grid(row=0, column=1, sticky="e")
        ttk.Button(self.right_buttons, text="📱 Emergency Contacts", command=self.open_phone_window, style="Contacts.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(self.right_buttons, text="🔔 Custom Alert Word", command=self.set_custom_word, style="Custom.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(self.right_buttons, text="🚨 EMERGENCY SOS", command=self.trigger_emergency, style="Emergency.TButton").pack(side=tk.LEFT, padx=5)
    def open_phone_window(self): PhoneNumberWindow(self.root)
    def set_custom_word(self):
        cw = simpledialog.askstring("Custom Alert Word", "Enter a custom word to trigger alerts:", parent=self.root)
        if cw: self.log(f"Custom alert word set to: {cw}")
    def update_time(self):
        self.time_label.config(text=datetime.now().strftime("%I:%M:%S %p")); self.root.after(1000, self.update_time)
    def start_monitoring(self):
        self.camera_running, self.is_recording, self.frames, self.model, self.last_sos, self.cooldown, self.fps = True, False, [], YOLO("yolov8n.pt"), 0, 3, 30
        self.camera_thread = threading.Thread(target=self.update_camera_feed, daemon=True); self.camera_thread.start()
        self.update_status("Monitoring", "Active"); self.log("Monitoring started."); self.start_btn.config(state=tk.DISABLED); self.stop_btn.config(state=tk.NORMAL)
    def stop_monitoring(self):
        self.camera_running = False
        if hasattr(self, 'camera_thread'): self.camera_thread.join(timeout=1.0)
        [l.configure(image='') for l in self.view_labels.values()]
        self.update_status("Monitoring", "Stopped"); self.log("Monitoring stopped."); self.start_btn.config(state=tk.NORMAL); self.stop_btn.config(state=tk.DISABLED)
    def setup_camera_frame(self, main):
        self.camera_frame = tk.Frame(main, bg="#222"); self.camera_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 5), pady=(0, 5)); main.grid_columnconfigure(1, weight=4)
        self.camera_frame.grid_columnconfigure(0, weight=1, uniform="col"); self.camera_frame.grid_columnconfigure(1, weight=1, uniform="col")
        self.camera_frame.grid_rowconfigure(0, weight=1, uniform="row"); self.camera_frame.grid_rowconfigure(1, weight=1, uniform="row")
        view_titles = {"Color": "Original Feed", "Gray": "Grayscale View", "Edges": "Edge Detection", "Infra": "Infrared View"}
        self.view_labels = {}
        for idx, (name, title) in enumerate(view_titles.items()):
            r, c = divmod(idx, 2)
            frame = tk.Frame(self.camera_frame, bg="#222"); frame.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            frame.grid_columnconfigure(0, weight=1); frame.grid_rowconfigure(0, minsize=20); frame.grid_rowconfigure(1, weight=1)
            ttk.Label(frame, text=title, foreground="#fff", background="#222", font=("Helvetica", 10, "bold")).grid(row=0, column=0, pady=(5,2))
            self.view_labels[name] = ttk.Label(frame); self.view_labels[name].grid(row=1, column=0)
    def update_camera_feed(self):
        mp_hands, fingers, missing = mp.solutions.hands, ["thumb", "index", "middle", "ring", "pinky"], []
        cap = cv2.VideoCapture(0); cap.set(3, 640); cap.set(4, 480); self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.7) as hands:
            while self.camera_running:
                ret, img = cap.read()
                if not ret:
                    continue
                img = cv2.flip(img, 1); ori = img.copy(); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); edges = cv2.Canny(gray, 70, 200)
                views = {"Color": ori.copy(), "Gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Edges": cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Infra": np.zeros_like(img)}
                views["Infra"][:, :, 1] = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                sos = False
                res = hands.process(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0].landmark
                    folded = [f for f in fingers if f not in missing and lm[{"thumb":4,"index":8,"middle":12,"ring":16,"pinky":20}[f]].y > lm[{"thumb":3,"index":6,"middle":10,"ring":14,"pinky":18}[f]].y]
                    if len(folded) == (5 - len(missing)): sos = True
                yres = self.model.predict(ori, verbose=False)[0]
                for b in yres.boxes:
                    if self.model.names[int(b.cls[0])] in ["knife", "scissors"] and float(b.conf[0]) > 0.4:
                        sos = True; x1, y1, x2, y2 = map(int, b.xyxy[0]); cv2.rectangle(views["Color"], (x1, y1), (x2, y2), (0, 0, 255), 2)
                if sos and time.time() - self.last_sos > self.cooldown:
                    self.last_sos = time.time(); self.trigger_emergency()
                if self.is_recording:
                    self.frames.append(ori.copy())
                    if time.time() - self.recording_start >= 5:
                        self.st.set(); time.sleep(0.2); audio = self.q.get(); merge_save(self.frames, audio); self.is_recording = False; self.frames = []
                for name, image in views.items():
                    if sos: cv2.putText(image, "!!! SOS !!!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); image = cv2.resize(image, (480, 360))
                    photo = ImageTk.PhotoImage(image=Image.fromarray(image)); self.view_labels[name].configure(image=photo); self.view_labels[name].image = photo
        cap.release()
    def trigger_emergency(self):
        self.log("🚨 EMERGENCY SOS TRIGGERED!"); self.update_status("SOS Detected", "YES")
        
        # Start beep sound in a separate thread
        def play_beep():
            for _ in range(3):  # Play 3 beeps
                winsound.Beep(1000, 1000)  # 1000Hz for 1 second
        threading.Thread(target=play_beep, daemon=True).start()
        
        if not self.is_recording:
            self.is_recording, self.frames, self.q, self.st = True, [], Queue(), threading.Event()
            threading.Thread(target=record_audio, args=(self.q, self.st), daemon=True).start(); self.recording_start = time.time()
        g = geocoder.ip("me")
        if g.ok: lat, lon = g.latlng; location = f"https://maps.google.com/?q={lat},{lon}"; self.update_status("Location", location); msg = f"🚨SOS! My location: {location}"
        else: msg = "🚨SOS! Location unavailable."; self.update_status("Location", "Unavailable")
        def send_to_all_contacts():
            try:
                with open("emergency_contacts.txt", "r") as f:
                    contacts = [line.strip().split("|") for line in f.readlines()]
                if contacts:
                    self.log("📱 Alerting emergency contacts...")
                    for name, number in contacts:
                        self.log(f"Sending alert to {name}")
                        try:
                            WhatsAppAutomation.send_message(name, msg, disable_failsafe=True)
                            self.log(f"Alert sent to {name}")
                        except Exception as e:
                            self.log(f"Failed to send alert to {name}: {str(e)}")
            except FileNotFoundError:
                self.log("⚠ No emergency contacts found!")
            threading.Thread(target=lambda: messagebox.showwarning("EMERGENCY", "🚨 SOS Alert Triggered!"), daemon=True).start()
        threading.Thread(target=send_to_all_contacts, daemon=True).start()
    def update_status(self, field, value): field in self.labels and self.labels[field].config(text=value)
    def log(self, message): self.log_box.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"); self.log_box.see(tk.END)

    def start_emotion_monitoring(self):
        def monitor():  # ✅ Define the function before using it
            self.monitoring = True
            print("[INFO] Started emotion monitoring every 5 seconds")

            while self.monitoring:
                try:
                    # Record audio for 5 seconds at 16kHz
                    duration = 5
                    fs = 16000
                    print("[INFO] Recording...")
                    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
                    sd.wait()
                    

                    # Convert to 1D numpy array
                    audio = np.squeeze(audio)

                    # Call your model
                    emotion, confidence = predict_emotion(audio)
                    print(f"[RESULT] Emotion: {emotion}, Confidence: {confidence:.2f}")

                except Exception as e:
                    print(f"[ERROR] During monitoring: {e}")

        # ✅ Start thread only after defining monitor()
        threading.Thread(target=monitor, daemon=True).start()

if __name__ == "__main__":
    try: root = tk.Tk(); app = SOSMonitorApp(root); root.mainloop()
    except Exception as e: print(f"Error starting application: {e}")