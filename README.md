# Monkey Gesture Detection

A real-time desktop application that detects hand gestures and displays corresponding monkey images.

## Features
- **Real-time Hand Detection**: Uses MediaPipe to track hand landmarks.
- **Gesture Recognition**: Detects 5 specific gestures:
    - Thumbs Up 👍
    - Peace Sign ✌️
    - Thinking 🤔
    - Pointing 👉
    - Shocked 😲
- **Interactive Feedback**: Overlays a funny monkey image for each gesture.
- **Visual Debugging**: Shows hand skeleton and detecting confidence.

## Tech Stack
- Python 3.9+
- OpenCV (Computer Vision)
- MediaPipe (Hand Tracking)
- NumPy (Vector Math)
- Pillow (Image Processing)

## Setup

**Important:** This project works best with **Python 3.9 to 3.11**. (Python 3.14 is currently too new for some libraries).

1.  **Create a virtual environment** (recommended):
    ```bash
    # If you have multiple python versions, specify one (e.g., 3.11)
    py -3.11 -m venv venv
    
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    python main.py
    ```

## Controls
- **'q'**: Quit the application.

## Troubleshooting
- Ensure your webcam is connected.
- If the application closes immediately, check the console for error messages.
- Ensure the `images/` directory contains the required monkey images.
