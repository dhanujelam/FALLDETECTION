# AI CCTV System (FallGuard Pro)

An advanced, AI-driven CCTV surveillance system built with Python, Flask, and Ultralytics YOLOv8 for pose estimation. The system monitors live camera feeds to provide real-time frame analysis including fall detection, SOS gesture recognition, and crowd management. It also features a physical hardware integration component, connecting directly to an Arduino for sounding a buzzer during emergency situations.

## Features

- **Fall Detection:** Uses YOLOv8 pose estimation to detect and trigger alerts for immediate fall events.
- **SOS Gesture Detection:** Recognizes specific distress hand gestures.
- **Crowd Management:** Keeps track of the number of people on screen and issues alerts for dangerous overcrowding events.
- **Hardware Integration:** Communicates over serial (`COM3`) to an Arduino to activate a physical alarm during an emergency.
- **Modern Web Dashboard:** Provides a clear UI to monitor the real-time camera feed, system logs, active statuses, and more.
- **Incident Database Logging:** Automatically records triggered security events.

## Prerequisites

- **Python 3.8+**
- **Hardware Requirements:** An Arduino Uno (must be connected to `COM3` on Windows by default) equipped with a buzzer circuit.
- **Webcam / Camera feed interface**

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd "AI CCTV System"
   ```

2. **Install all required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Arduino Setup:**
   Ensure your Arduino is plugged in and configured to listen at a baud rate of 9600. The code sends a `"1"` over serial to turn the alarm ON and a `"0"` to turn it OFF.

## Usage

1. Open a terminal and run the main application file:
   ```bash
   python app.py
   ```
2. Once the system initializes, open your web browser and navigate to:
   `http://127.0.0.1:5000`
3. Toggle features and monitor the feed from the web interface.

## Project Structure

- `app.py`: Entry point for the Flask web server, stream routing, and Arduino communication.
- `detection_engine.py`: Contains the logic handling computer vision (YOLOv8 pose).
- `tracking_engine.py`: Manages temporal tracking across subsequent frames.
- `alert_manager.py`: Responsible for archiving and logging incidents logic securely.
- `static/` & `templates/`: Assets and HTML templates for the web interface.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
