import cv2
import numpy as np
import mediapipe as mp
import time
import serial
import serial.tools.list_ports
import json

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

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

# Finger landmark indices
fingers = {
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

def get_landmark_coords(hand_landmarks, landmark_index):
    return hand_landmarks.landmark[landmark_index]

def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def length(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_angle(lm1, lm2, lm3):
    v1 = Vector(lm1.x - lm2.x, lm1.y - lm2.y, lm1.z - lm2.z)
    v2 = Vector(lm2.x - lm3.x, lm2.y - lm3.y, lm2.z - lm3.z)
    
    dot = dot_product(v1, v2)
    denom = length(v1.x, v1.y, v1.z) * length(v2.x, v2.y, v2.z)
    
    if denom == 0:
        return 0
    
    angle = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
    return angle

def calculate_finger_angle(hand_landmarks, a, b, c, d, root, thumb=False):
    if hand_landmarks is None:
        return None
    
    mcp = get_landmark_coords(hand_landmarks, a)
    pip = get_landmark_coords(hand_landmarks, b)
    dip = get_landmark_coords(hand_landmarks, c)
    tip = get_landmark_coords(hand_landmarks, d)
    
    if thumb:
        angle_2 = calculate_angle(mcp, pip, dip)
        angle_3 = calculate_angle(pip, dip, tip)
        return np.mean([angle_2, angle_3])
    else:
        angle_1 = calculate_angle(root, mcp, pip)
        angle_2 = calculate_angle(mcp, pip, dip)
        angle_3 = calculate_angle(pip, dip, tip)
        return np.mean([angle_1, angle_2, angle_3])

def calculate_angle_to_plane(lm1, lm2, lm3, lm4):
    v1 = Vector(lm1.x - lm2.x, lm1.y - lm2.y, lm1.z - lm2.z)
    v2 = Vector(lm1.x - lm3.x, lm1.y - lm3.y, lm1.z - lm3.z)
    
    normal = Vector(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    )
    
    reference = Vector(lm1.x - lm4.x, lm1.y - lm4.y, lm1.z - lm4.z)
    
    dot = dot_product(normal, reference)
    denom = length(normal.x, normal.y, normal.z) * length(reference.x, reference.y, reference.z)
    
    if denom == 0:
        return 0
    
    angle = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
    return np.clip(abs(90 - angle), 0, 90)

def calculate_distance_3d(lm1, lm2):
    """Calculate 3D distance between two landmarks"""
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    dz = lm1.z - lm2.z
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def calculate_thumb_mcp_bend(hand_landmarks):
    """
    Calculate thumb PIP+DIP joint bend ONLY.
    Since thumb has 2 joints but we can only measure 1 angle, multiply by 2.
    """
    if hand_landmarks is None:
        return 0
    
    # Get thumb landmarks
    thumb_mcp = get_landmark_coords(hand_landmarks, 2)   # MCP joint
    thumb_pip = get_landmark_coords(hand_landmarks, 3)   # PIP joint  
    thumb_dip = get_landmark_coords(hand_landmarks, 4)   # DIP joint (tip)
    
    if thumb_mcp is None or thumb_pip is None or thumb_dip is None:
        return 0
    
    # Calculate the angle at PIP joint: MCP-PIP-DIP
    pip_angle = calculate_angle(thumb_mcp, thumb_pip, thumb_dip)
    
    if pip_angle is None:
        return 0
    
    # Since we have 2 joints but can only measure 1 angle, multiply by 2
    scaled_angle = pip_angle * 2
    
    # Normalize to 0-180 range
    normalized_angle = int(np.round(np.clip(scaled_angle, 0, 180)))
    
    # Fix weird shit - if it equals 40, make it 0
    if normalized_angle == 40:
        normalized_angle = 0
    
    return normalized_angle

def calculate_thumb_opposition(hand_landmarks):
    """
    Calculate TRUE thumb opposition - how much the thumb crosses the palm.
    0 = rest position (beside index finger, not crossing palm)
    180 = full opposition (crossing palm to reach pinky side)
    """
    if hand_landmarks is None:
        return 0
    
    try:
        # Get key landmarks
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # Finger MCP joints to define palm width
        index_mcp = hand_landmarks.landmark[5]
        middle_mcp = hand_landmarks.landmark[9]
        ring_mcp = hand_landmarks.landmark[13]
        pinky_mcp = hand_landmarks.landmark[17]
        
        # Create palm coordinate system
        # Palm line from index MCP to pinky MCP
        palm_vector = Vector(
            pinky_mcp.x - index_mcp.x,
            pinky_mcp.y - index_mcp.y,
            pinky_mcp.z - index_mcp.z
        )
        
        # Vector from index MCP to thumb tip (this shows how far thumb crosses)
        thumb_cross_vector = Vector(
            thumb_tip.x - index_mcp.x,
            thumb_tip.y - index_mcp.y,
            thumb_tip.z - index_mcp.z
        )
        
        # Project thumb position onto palm width direction
        palm_length = length(palm_vector.x, palm_vector.y, palm_vector.z)
        if palm_length == 0:
            return 0
        
        # Normalize palm vector
        palm_unit = Vector(
            palm_vector.x / palm_length,
            palm_vector.y / palm_length,
            palm_vector.z / palm_length
        )
        
        # Calculate how far thumb crosses along palm width
        cross_distance = dot_product(thumb_cross_vector, palm_unit)
        
        # Normalize relative to palm width
        opposition_ratio = cross_distance / palm_length
        
        # Convert to 0-180 scale
        # 0 = no crossing (beside index)
        # 180 = full crossing (to pinky side)
        opposition_angle = np.clip(opposition_ratio * 180, 0, 180)
        
        return int(np.round(opposition_angle))
    except Exception as e:
        # If calculation fails, return a safe default
        return 0

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

            current_time = time.time()
            
            # Send only every 0.2 seconds
            if current_time - last_send_time >= 0.2:
                finger_angles = {}
                
                # Calculate angles for all fingers
                for finger_name, indices in fingers.items():
                    angle = calculate_finger_angle(
                        hand_landmarks,
                        indices[0], indices[1], indices[2], indices[3],
                        hand_landmarks.landmark[0]
                    )
                    # Normalize to 0-180 degrees
                    normalized_angle = int(np.round(np.clip(180 * angle / 79, 0, 180)))
                    finger_angles[finger_name] = normalized_angle
                
                # Apply 20-degree threshold rule: if bend is 20 or less, make it 0
                for finger_name in finger_angles:
                    if finger_angles[finger_name] <= 20:
                        finger_angles[finger_name] = 0
                
                # Thumb MCP bend (improved calculation - DIP+PIP only)
                thumb_bend = calculate_thumb_mcp_bend(hand_landmarks)
                
                # Thumb opposition - palm crossing (send as thumb_rot for ESP32 compatibility)
                thumb_opposition = calculate_thumb_opposition(hand_landmarks)
                
                # Apply 20-degree threshold rule to thumb as well
                # Also catch the weird 40° straight thumb value
                if thumb_bend <= 20 or thumb_bend == 40:
                    thumb_bend = 0
                if thumb_opposition <= 20:
                    thumb_opposition = 0
                
                finger_angles['thumb_bend'] = thumb_bend
                finger_angles['thumb_rot'] = thumb_opposition  # ESP32 expects 'thumb_rot'
                
                # Send to ESP32 if connected
                if serial_connected and ser:
                    try:
                        json_data = json.dumps(finger_angles)
                        ser.write((json_data + "\n").encode())
                        
                        # Print current values for debugging
                        print(f"\rIndex:{finger_angles.get('index', 0):3d} Middle:{finger_angles.get('middle', 0):3d} Ring:{finger_angles.get('ring', 0):3d} Pinky:{finger_angles.get('pinky', 0):3d} | ThumbBend:{finger_angles.get('thumb_bend', 0):3d} ThumbOpp:{finger_angles.get('thumb_rot', 0):3d}", end="")
                        
                        # Read response from ESP32
                        if ser.in_waiting:
                            line = ser.readline().decode(errors="ignore").strip()
                            if line:
                                print(f"\nESP32: {line}")
                    except Exception as e:
                        print(f"Serial error: {e}")
                else:
                    # Print values even without ESP32 for testing
                    print(f"\rIndex:{finger_angles.get('index', 0):3d} Middle:{finger_angles.get('middle', 0):3d} Ring:{finger_angles.get('ring', 0):3d} Pinky:{finger_angles.get('pinky', 0):3d} | ThumbBend:{finger_angles.get('thumb_bend', 0):3d} ThumbOpp:{finger_angles.get('thumb_rot', 0):3d}", end="")
                
                last_send_time = current_time

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
    print("Serial connection closed.")
