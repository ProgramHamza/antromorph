import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import serial
import struct
import json

class Vector:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set up Matplotlib for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

serial_port = 'COM15'  # Update with your port
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Function to extract landmark coordinates
def get_landmark_coords(hand_landmarks, landmark_index):
    lm = hand_landmarks.landmark[landmark_index]
    return lm

def dot_product(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def unit_vector(x, y, z):
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return [x / magnitude, y / magnitude, z / magnitude] if magnitude else [0, 0, 0]

def length(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def calculate_angle(lm1,lm2,lm3):
    v1 = Vector(lm1.x-lm2.x,lm1.y-lm2.y,lm1.z-lm2.z)
    v2 = Vector(lm2.x-lm3.x,lm2.y-lm3.y,lm2.z-lm3.z)

    angle = np.degrees(np.arccos(dot_product(v1,v2)/(length(v1.x,v1.y,v1.z)*length(v2.x,v2.y,v2.z))))
    return angle

def calculate_angle_3(hand_landmarks,a,b,c,d,root,thumb = False):
    if hand_landmarks is None:
        return None, None
    if thumb == False:
        mcp = get_landmark_coords(hand_landmarks, a)  # MCP joint
        pip = get_landmark_coords(hand_landmarks, b)  # PIP joint
        dip = get_landmark_coords(hand_landmarks, c)  # DIP joint
        tip = get_landmark_coords(hand_landmarks, d)  # Fingertip

        angle_1 = calculate_angle(root,mcp,pip)
        angle_2 = calculate_angle(mcp,pip,dip)
        angle_3 = calculate_angle(pip,dip,tip)
        vysledok = np.mean([angle_1,angle_2,angle_3])
        return vysledok
    else:
        mcp = get_landmark_coords(hand_landmarks, a)  # MCP joint
        pip = get_landmark_coords(hand_landmarks, b)  # PIP joint
        dip = get_landmark_coords(hand_landmarks, c)  # DIP joint
        tip = get_landmark_coords(hand_landmarks, d)  # Fingertip

        angle_2 = calculate_angle(mcp,pip,dip)
        angle_3 = calculate_angle(pip,dip,tip)
        vysledok = np.mean([angle_2,angle_3])
        return vysledok
fingers = {'index' : [5,6,7,8],
'middle' : [9,10,11,12],
'ring' : [13,14,15,16],
# 'pinky' : [17,18,19,20],
}

def calculate_angle_to_plane(lm1, lm2, lm3,lm4):
    v1 = Vector(lm1.x - lm2.x, lm1.y - lm2.y, lm1.z - lm2.z)
    v2 = Vector(lm1.x - lm3.x, lm1.y - lm3.y, lm1.z - lm3.z)

    normal = Vector(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    )

    reference = Vector(lm1.x - lm4.x, lm1.y - lm4.y, lm1.z - lm4.z)

    angle = np.degrees(np.arccos(dot_product(normal, reference) / (length(normal.x, normal.y, normal.z) * length(reference.x, reference.y, reference.z))))
    return np.clip(abs(90-angle),0,90)

# The original finger function you provided
def finger(stx, sty, stz, segment_length, degrees, bending_angle, thumb=False):

    ax.clear()
    k = segment_length
    bend = np.radians(degrees)  # Convert to radians
    bend1 = np.radians(bending_angle)  # Convert to radians

    def unit_vector(x, y, z):
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        return [x / magnitude, y / magnitude, z / magnitude] if magnitude else [0, 0, 0]

    # First knuckle movement
    a = -k * np.cos(bend)
    b = k * np.sin(bend)

    def new_x(start_x, prev_x, start_y, prev_y):
        return start_x + a * unit_vector(start_x - prev_x, start_y - prev_y, 0)[0] + b * unit_vector(prev_y - start_y, start_x - prev_x, 0)[0]

    def new_y(start_x, prev_x, start_y, prev_y):
        return start_y + a * unit_vector(start_x - prev_x, start_y - prev_y, 0)[1] + b * unit_vector(prev_y - start_y, start_x - prev_x, 0)[1]

    c = new_x(stx + a, stx, b + sty, sty)
    d = new_y(stx + a, stx, b + sty, sty)
    e = new_x(c, stx + a, d, sty + b)
    f = new_y(c, stx + a, d, sty + b)

    if thumb:
        body = [
            [stx, sty, stz],
            [stx + a, sty + b, stz],
            [c, d, stz],
        ]
    else:
        body = [
            [stx, sty, stz],
            [stx + np.cos(bend1) * a, sty + b, stz + k * np.sin(bend1)],
            [c * np.cos(bend1), d, stz + (c - stx) * np.sin(bend1)],
            [e * np.cos(bend1), f, stz + (e - stx) * np.sin(bend1)],
        ]
        
     
    x = [element[0] for element in body]
    y = [element[1] for element in body]
    z = [element[2] for element in body]

    ax.scatter(x, y, z)
    ax.plot(x, y, z)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Hand Visualization")
    plt.grid()
    plt.draw()
a = []
# Open webcam
cap = cv2.VideoCapture(0)
previous_angles = {'index': 0, 'middle': 0, 'ring': 0, 'palecbend': 0, 'palecrot': 0}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            figner_dict = {}

            for i in fingers:
                angle = calculate_angle_3(
                    hand_landmarks,
                    fingers[i][0],
                    fingers[i][1],
                    fingers[i][2],
                    fingers[i][3],
                    hand_landmarks.landmark[0]
                ) 
                value_for_servo = int(np.round(np.clip(180 * angle / 79, 0, 180)))
                figner_dict[i] = value_for_servo-previous_angles[i]


            # Thumb
            anglethumb = calculate_angle_3(
                hand_landmarks,
                1, 2, 3, 4,
                hand_landmarks.landmark[0],
                thumb=True
            )
            figner_dict['palecbend'] = int(np.round(np.clip(180 * anglethumb / 79, 0, 180)))-previous_angles['palecbend']

            thumb_root_angle = calculate_angle_to_plane(
                hand_landmarks.landmark[0], hand_landmarks.landmark[5],
                hand_landmarks.landmark[17], hand_landmarks.landmark[2]
            )
            thumb_root_angle2 = calculate_angle_to_plane(
                hand_landmarks.landmark[0], hand_landmarks.landmark[5],
                hand_landmarks.landmark[17], hand_landmarks.landmark[1]
            )
            thumb_root_mean = np.mean([thumb_root_angle, thumb_root_angle2])
            
            if thumb_root_mean > 15:
                if previous_angles['palecrot'] != 160:
                    figner_dict['palecrot'] = 160 
                else:
                    figner_dict['palecrot'] = 0
            else:
                if previous_angles['palecrot'] != -160:
                    figner_dict['palecrot'] = -160 
                else:
                    figner_dict['palecrot'] = 0


            # --- SERIAL SEND ---
            json_data = json.dumps(figner_dict)
            ser.write((json_data + "\n").encode())  # newline = end of message
            print("Sent:", json_data)
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print("ESP32:", line)
            previous_angles = figner_dict.copy()
            time.sleep(0.25)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()