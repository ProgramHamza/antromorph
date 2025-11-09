from machine import Pin, PWM
import time
import sys
import json

class Servo:
    # these defaults work for the standard TowerPro SG90
    __servo_pwm_freq = 50
    __min_u10_duty = 26 - 0 # offset for correction
    __max_u10_duty = 123- 0  # offset for correction
    min_angle = 0
    max_angle = 180
    current_angle = 0.001


    def __init__(self, pin, name="servo"):
        self.name = name  # Store servo name for error messages
        self.__initialise(pin)


    def update_settings(self, servo_pwm_freq, min_u10_duty, max_u10_duty, min_angle, max_angle, pin):
        self.__servo_pwm_freq = servo_pwm_freq
        self.__min_u10_duty = min_u10_duty
        self.__max_u10_duty = max_u10_duty
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.__initialise(pin)


    def move(self, angle):
        # round to 2 decimal places, so we have a chance of reducing unwanted servo adjustments
        target_angle = round(angle, 2)
        
        # If same angle, no need to move
        if target_angle == self.current_angle:
            return
        
        # For fast movement, move directly to target
        self.current_angle = target_angle
        duty_u10 = self.__angle_to_u10_duty(self.current_angle)
        self.__motor.duty(duty_u10)
        
        # Small delay to prevent servo jitter
        time.sleep_ms(5)

    def __angle_to_u10_duty(self, angle):
        return int((angle - self.min_angle) * self.__angle_conversion_factor) + self.__min_u10_duty


    def __initialise(self, pin):
        self.current_angle = -0.001
        self.__angle_conversion_factor = (self.__max_u10_duty - self.__min_u10_duty) / (self.max_angle - self.min_angle)
        self.__motor = PWM(Pin(pin))
        self.__motor.freq(self.__servo_pwm_freq)


# Pin setup for all fingers
PINS = {
    'index': 12,
    'middle': 11,
    'ring': 10,
    'pinky': 7,         # pinky finger
    'thumb_rot': 6,     # thumb rotation
    'thumb_bend': 4     # thumb bending
}

# Initialize all servos
servos = {}
for name, pin in PINS.items():
    servos[name] = Servo(pin=pin, name=name)  # Pass name for error messages
    servos[name].move(0)  # Initialize to 0 degrees
    print(f"{name} initialized on pin {pin}")

print("All servos ready. Waiting for commands...")

while True:
    try:
        # Read data from serial input
        line = sys.stdin.readline().strip()
        if not line:
            continue
        
        # Try parsing as JSON first (for multiple fingers)
        try:
            commands = json.loads(line)
            # Handle JSON object with multiple finger commands
            if isinstance(commands, dict):
                # Move all servos in parallel (issue all commands at once)
                moved_fingers = []
                for finger_name, degrees in commands.items():
                    if finger_name in servos:
                        degrees = int(degrees)
                        servos[finger_name].move(degrees)  # PWM signal sent immediately
                        moved_fingers.append(f"{finger_name}: {degrees}°")
                    else:
                        print(f"Unknown finger: {finger_name}")
                
                # Print all movements at once
                if moved_fingers:
                    print(" | ".join(moved_fingers))
            else:
                print("Invalid JSON format")
        
        except json.JSONDecodeError:
            # If not JSON, try parsing as single number (for index finger only)
            try:
                degrees = int(line)
                servos['index'].move(degrees)
                print(f"index: {degrees}°")
            except ValueError:
                print(f"Invalid input: {line}")
        
        time.sleep(0.01)  # Reduced from 0.05 for faster response
        
    except Exception as e:
        print(f"Error: {e}")






