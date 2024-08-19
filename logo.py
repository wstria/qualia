import time
import sys

def text_streaming_effect(text, delay=0.05):
    for line in text:
        for char in line:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write('\n')
        sys.stdout.flush()

def display_logo():
    logo = [
        "                             /\_ \      __              ",
        "      __     __  __      __  \//\ \    /\_\       __     ",
        "    /'__`\  /\ \/\ \   /'__`\  \ \ \   \/\ \    /'__`\   ",
        "   /\ \L\ \ \ \ \_\ \ /\ \L\.\_ \_\ \_  \ \ \  /\ \L\.\_ ",
        "   \ \___, \ \ \____/ \ \__/.\_\ /\____\ \ \_\ \ \__/.\_\ ",
        "    \/___/\ \ \/___/   \/__/\/_/ \/____/  \/_/  \/__/\/_/ ",
        "         \ \_\                                      ",
        "          \/_/ ",
    ]
    
    text_streaming_effect(logo, delay=0.008)
