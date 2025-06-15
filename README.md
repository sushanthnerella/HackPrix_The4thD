# IndriyaRaksha: Women Safety Guardian

## Overview
IndriyaRaksha is an integrated women safety and emergency response system. It combines real-time emotion and audio monitoring, video surveillance, WhatsApp automation, and a smart AI chatbot to detect distress, trigger alerts, and assist users in emergencies. The system leverages computer vision, speech emotion recognition, location tracking, and automated messaging to provide a comprehensive safety net.

## Features
- **Emotion Detection & Audio Monitoring**: Continuously analyzes live audio for distress emotions (fear, sadness) and abnormal decibel levels using an LSTM model and decibel meter.
- **SOS Trigger & Emergency Alerts**: Detects custom trigger words (e.g., "help") in speech using OpenAI Whisper, and sends SOS alerts with live location to emergency contacts via WhatsApp.
- **Camera Surveillance**: Monitors video feed using YOLOv8 for suspicious activity and integrates with the main interface.
- **Sensor Integration**: Supports additional sensor-based triggers for enhanced safety.
- **AI Chatbot**: Provides a voice-enabled assistant for information, news, jokes, location sharing, and WhatsApp contact management.
- **WhatsApp Automation**: Automates sending messages, calls, and location sharing to contacts using GUI automation.
- **Contact Management**: Add, remove, and manage emergency contacts through a user-friendly interface.
- **Logging & UI**: Modern Tkinter-based GUI with real-time status, logs, and multi-tab navigation.

## Directory Structure
- `interface.py`, `cameraInterface.py`, `camint4.py`: Main GUI applications integrating all features.
- `emotion_detection.py`, `emotion_monitor.py`: Speech emotion recognition logic and monitoring loop.
- `dbmeter.py`: Decibel level calculation from audio.
- `whatsapp_automation.py`: WhatsApp automation logic (messaging, calls, location sharing).
- `Final_ChatBot_Int.py`: AI chatbot with voice, news, weather, jokes, and WhatsApp integration.
- `sensor_component.py`, `sensor.py`: Sensor integration for additional triggers.
- `locationsearch.py`: Location lookup and geocoding.
- `utils.py`, `styles.py`: Utility functions and UI styling.
- `phone_window.py`: Phone number input dialog.
- `emergency_contacts.txt`: List of emergency contacts (Name|Number).
- `AudioWAV/`: Sample audio files for testing.
- `vids/`: Saved video/audio recordings.
- `liquid_model.pth`, `lstm_model2.pth1`, `yolov8n.pt`: Pretrained models for emotion and video analysis.

## Setup & Requirements
- **Python 3.8+**
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  # or manually:
  pip install tkinter opencv-python numpy sounddevice mediapipe ultralytics pillow pyautogui geocoder whisper pyttsx3 requests beautifulsoup4
  ```
- Place model files (`liquid_model.pth`, `lstm_model2.pth1`, `yolov8n.pt`) in the project root.
- (Optional) Add your emergency contacts to `emergency_contacts.txt` in the format: `Name|Number`

## Usage
- Run the main interface:
  ```bash
  python interface.py
  # or
  python cameraInterface.py
  # or
  python camint4.py
  ```
- Use the GUI to start monitoring, manage contacts, and interact with the chatbot.
- The system will automatically monitor for distress and trigger alerts as needed.

## Main Components
- **Emotion & Audio Monitoring**: Uses microphone input to detect emotions and abnormal sound levels.
- **Video Surveillance**: Uses webcam and YOLOv8 for real-time video analysis.
- **WhatsApp Automation**: Uses pyautogui to control WhatsApp Desktop for messaging/calls.
- **AI Chatbot**: Voice/text assistant for information, news, jokes, and emergency help.
- **Location Tracking**: Uses geocoder to fetch and share live location.

## Authors
- The 4th Dimension (HackPrix Team)

## License
This project is for educational and hackathon purposes only. See LICENSE file if present.

---

**Note:**
- WhatsApp automation requires WhatsApp Desktop and may need screen coordinate adjustments for your system.
- Some features require internet access (location, news, weather, chatbot).
- For best results, run on Windows/Linux with all dependencies installed.


