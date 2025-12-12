import cv2
import numpy as np
import mediapipe as mp
import time
import serial
import serial.tools.list_ports
import json

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Automatic Serial Port Detection
def find_esp32_port():
    """Automatically detect ESP32 COM port"""
    ports = serial.tools.list_ports.comports()
    
    # Try COM22 first (preferred)
    for port in ports:
        if 'COM22' in port.device:
            try:
                ser = serial.Serial(port.device, 115200, timeout=1)
                print(f"Connected to ESP32 on {port.device}")
                return ser
            except:
                pass
    
    # Try other COM ports
    for port in ports:
        # Look for common ESP32 identifiers
        if any(keyword in port.description.upper() for keyword in ['CP210', 'CH340', 'USB', 'SERIAL']):
            try:
                ser = serial.Serial(port.device, 115200, timeout=1)
                print(f"Connected to ESP32 on {port.device} ({port.description})")
                return ser
            except:
                continue
    
    print("Warning: Could not find ESP32. Available ports:")
    for port in ports:
        print(f"  - {port.device}: {port.description}")
    return None

# Connect to ESP32
ser = find_esp32_port()
serial_connected = ser is not None

# Finger indices for index finger
fingers = {'index': [5, 6, 7, 8]}

def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def length(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_angle(lm1, lm2, lm3):
    v1 = [lm1.x - lm2.x, lm1.y - lm2.y, lm1.z - lm2.z]
    v2 = [lm2.x - lm3.x, lm2.y - lm3.y, lm2.z - lm3.z]
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    denom = length(*v1) * length(*v2)
    if denom == 0:
        return 0
    return np.degrees(np.arccos(dot / denom))

def calculate_finger_angle(hand_landmarks, a, b, c, d, root):
    if hand_landmarks is None:
        return None
    mcp = hand_landmarks.landmark[a]
    pip = hand_landmarks.landmark[b]
    dip = hand_landmarks.landmark[c]
    tip = hand_landmarks.landmark[d]
    angle_1 = calculate_angle(root, mcp, pip)
    angle_2 = calculate_angle(mcp, pip, dip)
    angle_3 = calculate_angle(pip, dip, tip)
    return np.mean([angle_1, angle_2, angle_3])

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting hand tracking... press 'q' to quit.")
last_send_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            angle = calculate_finger_angle(
                hand_landmarks,
                fingers['index'][0],
                fingers['index'][1],
                fingers['index'][2],
                fingers['index'][3],
                hand_landmarks.landmark[0]
            )

            degrees = int(np.round(np.clip(180 * angle / 79, 0, 180)))
            
            # Send only every 0.2 seconds
            current_time = time.time()
            if current_time - last_send_time >= 0.2:
                # Send to ESP32 if connected
                if serial_connected and ser:
                    try:
                        ser.write((str(degrees) + "\n").encode())
                        
                        # Read response from ESP32
                        if ser.in_waiting:
                            line = ser.readline().decode(errors="ignore").strip()
                            if line:
                                print(line)
                    except Exception as e:
                        print(f"Serial error: {e}")
                
                last_send_time = current_time

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
    print("Serial connection closed.")
