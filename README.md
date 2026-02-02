# Monkey Gesture Detection

A real-time webcam-based gesture recognition application that triggers specific reaction images based on user gestures. Built with OpenCV and MediaPipe.

## Features

- **Real-time Detection**: Uses MediaPipe Hands and Face Mesh for low-latency recognition.
- **Smart Logic**: Distinguishes between similar gestures (e.g., "Think" vs "Idea") using geometric heuristics.
- **Visual Feedback**: visualizes hand landmarks, face targets, and recognition status.

## Installation

1. **Prerequisites**
   - Python 3.7+
   - A webcam

2. **Install Dependencies**
   ```bash
   pip install opencv-python mediapipe numpy
   ```

## Usage

1. Navigate to the project directory:
   ```bash
   cd gesture-detection
   ```

2. Run the application:
   ```bash
   python main.py
   ```

3. **Controls**:
   - Press `q` to quit the application.

## Supported Gestures

The system recognizes the following gestures and displays a corresponding image:

| Gesture | Action / Description |
| :--- | :--- |
| **Default** | No specific gesture detected. |
| **Think** | Place your index finger to your temple or near your mouth. <br>_(Requires index finger extended, others curled, hand close to face)_ |
| **Idea** | Raise your index finger vertically. <br>_(Requires strict vertical index finger, thumb tucked)_ |
| **Mad** | Fold your arms across your chest (make fists). <br>_(Requires wrists crossed low, fingers curled)_ |
| **Scared** | **Option 1**: Raise both hands open near your face (Home Alone style). <br>**Option 2**: Clasp hands together below your neck. |
| **Tongue** | Stick your tongue out. <br>_(Requires mouth open and tongue color detection)_ |
| **Double** | Two faces detected in the frame. |

## Troubleshooting

- **Lighting**: Ensure your face and hands are well-lit for accurate detection.
- **Camera**: If the camera doesn't open, check if another application is using it.
- **Performance**: The app is optimized for performance (30 FPS target), but older hardware may experience lag.
