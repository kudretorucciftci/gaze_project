# Calibration-Free Gaze Tracking System

This project is a system that uses a webcam to track the user's eye movements in real-time and moves the mouse cursor on the screen accordingly. It does not require any calibration process and offers advanced computer control features such as left-click, right-click, zoom in, zoom out, and smooth page scrolling with various eye gestures (blinking, squinting, focusing).

## ✨ Features

- **Real-Time Gaze Tracking:** Works with a standard webcam.
- **Calibration-Free:** No need for lengthy user-specific calibration sessions.
- **Smart Cursor Correction:** Improves cursor control by learning small deviations in the user's gaze over time (`implicit_bias`).
- **Advanced Eye Gesture Control:**
  - **Left Click:** With a normal (bilateral) short blink.
  - **Right Click:** By blinking only the right eye.
  - **Zoom In:** By squinting both eyes.
  - **Zoom Out:** By a long blink (holding eyes closed for longer than 0.5 seconds).
  - **Smooth Page Scrolling:** By focusing on the top or bottom edges of the screen.

## 🚀 Installation

Follow the steps below to run the project on your local machine.

### Requirements

- Python 3.x
- `pip` (Python package manager)

### Steps

1. **Clone the Project:**
   ```sh
   git clone https://github.com/kudretorucciftci/gaze_project.git
   cd gaze_project
   ```

2. **Trained Model and Dataset:**
   The trained model file (`mpiigaze_finetuned_v2.keras`) required by the project is already in this repository.
   If you want to download the dataset used for fine-tuning the model:
   - **Download dataset:** [Gaze Dataset](https://drive.google.com/file/d/1F-DPjKiTrWjcpQ4Pguj3wMl9axEx06x9/view?usp=drive_link)

3. **Create and Activate a Python Virtual Environment:**
   This project requires specific library versions. It is highly recommended to use a virtual environment to avoid conflicts with system-wide packages.
   ```sh
   # Windows
   python -m venv gaze_final
   .\gaze_final\Scripts\activate
   ```

4. **Install Dependencies:**
   All necessary libraries are listed in the `requirements.txt` file. Install them using the following command:
   ```sh
   pip install -r requirements.txt
   ```

## 🏃‍♀️ Usage

After installation, use the following command to run the main application:
```sh
python gaze_tracking.py
```
When the application starts, your webcam will open and a window will appear on the screen. Your cursor will start following your eye movements. To close the application, simply press the `ESC` key while the camera window is active.

### Eye Gesture Commands

- **Left Click:** Move the cursor where you want and perform a short **bilateral blink**.
- **Right Click:** Move the cursor where you want and perform a short **right-eye blink**.
- **Zoom In:** Move the cursor to the window you want to zoom and **squint your eyes**. This simulates the `Ctrl + Mouse Wheel Up` command.
- **Zoom Out:** Move the cursor to the window you want to zoom out and **keep your eyes closed for more than 0.5 seconds**. This simulates the `Ctrl + Mouse Wheel Down` command.
- **Page Scrolling:** Move the cursor to the window you want to scroll.
  - **To Scroll Down:** Move your gaze to the **bottom edge** of the screen and hold for about 0.6 seconds.
  - **To Scroll Up:** Move your gaze to the **top edge** of the screen and hold for about 0.6 seconds.
  To stop scrolling, move your gaze away from the edge.

### Camera Selection
The project uses the first camera in the system by default (usually ID 0). If you are using your phone (e.g. via applications like **iVCam**) or another external camera, you may need to change the `0` value in the `cap = cv2.VideoCapture(0)` line in `gaze_tracking.py` to the device ID of your camera (`0`, `1`, `2`, etc.).

## 🛠️ Method and Model

This project combines several different technologies:

- **Face and Eye Detection:** **Google Mediapipe** library is used to detect facial and eye landmarks in real-time.
- **Gaze Estimation:** The image captured from the eye region is processed by a **TensorFlow/Keras** model (`mpiigaze_finetuned_v2.keras`) pre-trained on the **MPIIGaze** dataset and then fine-tuned with user data. This model estimates the gaze direction (pitch and yaw angles) from the eye image.
- **Cursor Control:** Predictions from the model pass through a series of filtering and smoothing processes to ensure smooth movement of the mouse cursor.

## ⚙️ Configuration

You can adjust the sensitivity of all control mechanisms according to your personal preference by editing the following variables inside the `gaze_tracking.py` file:

```python
def main():
    # ...
    # --- Smooth Scrolling Settings ---
    SCROLL_ZONE_HEIGHT = 70  # Height of active zone at top/bottom of screen (pixels)
    SCROLL_ACTIVATION_DWELL = 0.6  # Dwell time to start scrolling (seconds)

    # --- Eye Gesture Action Settings ---
    SQUINT_THRESHOLD = 0.019 # Squint threshold (higher value = more sensitive)
    BLINK_THRESHOLD = 0.012  # Blink threshold (smaller value = more closed eye)
    ACTION_COOLDOWN = 0.8    # General cooldown between actions
    LONG_BLINK_DURATION = 0.6 # Minimum duration for long blink (seconds)
    # ...
```

- **Scrolling Settings:**
    - `SCROLL_ZONE_HEIGHT`: Sets the thickness of the edge bar that triggers scrolling.
    - `SCROLL_ACTIVATION_DWELL`: Sets how long you need to wait at the edge for scrolling to start.
- **Action Settings:**
    - `SQUINT_THRESHOLD` and `BLINK_THRESHOLD`: Balances the sensitivity between squinting and blinking.
    - `ACTION_COOLDOWN`: Sets the minimum time between two commands.
    - `LONG_BLINK_DURATION`: Sets the minimum duration for a blink to be considered "long".