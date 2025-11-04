import machine
import time
import sys
import json

# Servo setup (continuous rotation)
index = machine.PWM(machine.Pin(15), freq=50)
middle = machine.PWM(machine.Pin(18), freq=50)
ring = machine.PWM(machine.Pin(19), freq=50)
palecrot = machine.PWM(machine.Pin(21), freq=50)
palecbend = machine.PWM(machine.Pin(17), freq=50)  # changed pin to avoid conflict

servos = {
    'index': index,
    'middle': middle,
    'ring': ring,
    'palecbend': palecbend,
    'palecrot': palecrot
}

def stop(pwm):
    pwm.duty(77)
    time.sleep(0.01)


def rot(degrees,pwm1):
    if degrees > 0:
        time1 = abs((0.5*degrees)/290)
        if time1 > 0.25:
            time1 = 0.25
        pwm1.duty(120)
        time.sleep(time1)
        stop(pwm1)
    else:
        time1 = abs((0.5*degrees)/300)
        if time1 > 0.25:
            time1 = 0.25
            print('halo')
        pwm1.duty(30)
        time.sleep(time1)
        stop(pwm1)


while True:
    try:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        # Try parsing JSON
        try:
            commands = json.loads(line)
        except:
            print("Invalid JSON:", line)
            continue

        for name, deg in commands.items():
            if name in servos:
                print("Rotating", name, "by", deg)
                rot(deg, servos[name])

    except Exception as e:
        print("Error:", e)




