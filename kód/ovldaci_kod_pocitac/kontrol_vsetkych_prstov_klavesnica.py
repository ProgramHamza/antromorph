import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import serial
import serial.tools.list_ports
import struct
import keyboard
import math
import json
import threading

def find_serial_port():
    """Find and return the first available serial port"""
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("No serial ports found!")
        return None
    
    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i+1}. {port.device} - {port.description}")
    
    # Try to find ESP32 or Arduino-like device
    for port in ports:
        if any(keyword in port.description.lower() for keyword in ['usb', 'serial', 'ch340', 'cp210', 'ftdi', 'esp32', 'arduino']):
            print(f"Auto-selected: {port.device} - {port.description}")
            return port.device
    
    # If no specific device found, use the first available port
    selected_port = ports[0].device
    print(f"Using first available port: {selected_port}")
    return selected_port

def main():
    BAUD_RATE = 115200

    # Find available serial port
    SERIAL_PORT = find_serial_port()
    if SERIAL_PORT is None:
        print("No serial port available. Please connect your device and try again.")
        return

    # Set up the serial connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")
    except serial.SerialException as e:
        print(f"Failed to connect to {SERIAL_PORT}: {e}")
        return
    
    # Current angles for all fingers (0-180 degrees)
    current_angles = {
        'index': 0,      # a/d keys
        'middle': 0,     # s/w keys  
        'ring': 0,       # q/e keys
        'pinky': 0,      # j/l keys
        'thumb_rot': 0,  # k/i keys
        'thumb_bend': 0  # o/p keys
    }
    
    # Key mapping to fingers and directions
    key_to_finger = {
        'a': ('index', 'forward'),     # index finger forward
        'd': ('index', 'backward'),    # index finger backward
        's': ('middle', 'forward'),    # middle finger forward
        'w': ('middle', 'backward'),   # middle finger backward
        'q': ('ring', 'forward'),      # ring finger forward
        'e': ('ring', 'backward'),     # ring finger backward
        'j': ('pinky', 'forward'),     # pinky finger forward
        'l': ('pinky', 'backward'),    # pinky finger backward
        'k': ('thumb_rot', 'forward'), # thumb rotation forward
        'i': ('thumb_rot', 'backward'), # thumb rotation backward
        'o': ('thumb_bend', 'forward'), # thumb bend forward
        'p': ('thumb_bend', 'backward') # thumb bend backward
    }
    
    # Track which keys are currently pressed
    pressed_keys = set()
    last_update_time = time.time()
    
    def send_angles():
        """Send current angles as JSON to ESP32"""
        try:
            json_data = json.dumps(current_angles)
            ser.write((json_data + '\n').encode('utf-8'))
            print(f'Sent: {json_data}')
        except Exception as e:
            print(f'Error sending data: {e}')
    
    def update_angles():
        """Update angles for currently pressed keys (10 degrees per second)"""
        nonlocal last_update_time
        current_time = time.time()
        time_delta = current_time - last_update_time
        
        if time_delta >= 1.0:  # Update every second
            angle_change = 10  # 10 degrees per second
            angles_changed = False
            
            for key in pressed_keys:
                if key in key_to_finger:
                    finger, direction = key_to_finger[key]
                    
                    if direction == 'forward':
                        new_angle = min(180, current_angles[finger] + angle_change)
                    else:  # backward
                        new_angle = max(0, current_angles[finger] - angle_change)
                    
                    if new_angle != current_angles[finger]:
                        current_angles[finger] = new_angle
                        angles_changed = True
                        print(f'{finger} {direction}: {new_angle}°')
            
            if angles_changed:
                send_angles()
            
            last_update_time = current_time
    
    def check_key_states():
        """Check which keys are currently pressed"""
        nonlocal pressed_keys
        new_pressed_keys = set()
        
        for key in key_to_finger.keys():
            if keyboard.is_pressed(key):
                new_pressed_keys.add(key)
        
        # Handle newly pressed keys
        newly_pressed = new_pressed_keys - pressed_keys
        for key in newly_pressed:
            finger, direction = key_to_finger[key]
            print(f'Key {key} pressed - {finger} {direction}')
        
        # Handle newly released keys
        newly_released = pressed_keys - new_pressed_keys
        for key in newly_released:
            finger, direction = key_to_finger[key]
            print(f'Key {key} released - {finger} {direction}')
        
        pressed_keys = new_pressed_keys
    
    print("Servo control ready!")
    print("Controls:")
    print("  a/d - Index finger forward/backward")
    print("  s/w - Middle finger forward/backward") 
    print("  q/e - Ring finger forward/backward")
    print("  j/l - Pinky finger forward/backward")
    print("  k/i - Thumb rotation forward/backward")
    print("  o/p - Thumb bend forward/backward")
    print("  SPACE - Quit")
    print("\nHold keys for continuous 10°/sec movement")
    
    # Send initial position
    send_angles()
    
    while True:
        check_key_states()
        update_angles()
        time.sleep(0.1)  # Check every 100ms for responsive key detection

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
            break
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            print("Please check your device connection and try again.")
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)  # Wait before retrying
        
        if keyboard.is_pressed("space"):
            print("Space pressed - exiting")
            break