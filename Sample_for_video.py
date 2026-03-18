import cv2
import numpy as np
import os

def generate_morse_video():
    MORSE_DICT = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
        '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
        '9': '----.', '0': '-----', ' ': '/'
    }

    # 1. Get Input
    text = input("Enter the sentence for your Morse Video: ").upper()
    filename = "morse_video.mp4"
    
    # 2. Video Settings
    width, height = 640, 480
    fps = 30 
    unit_frames = 6 # Timing unit
    
    # Define Video Writer - Using 'mp4v' for Windows compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def write_frames(is_on, duration_units):
        color = 255 if is_on else 0
        # Create the frame
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        for _ in range(duration_units * unit_frames):
            out.write(frame)

    print("Generating video... Please wait.")

    # 3. Main Logic Loop
    for char in text:
        if char == ' ':
            write_frames(False, 7) # Gap between words
            continue
            
        if char in MORSE_DICT:
            symbols = MORSE_DICT[char]
            for i, symbol in enumerate(symbols):
                # Light ON
                duration = 1 if symbol == '.' else 3
                write_frames(True, duration)
                
                # Gap between dots/dashes (OFF)
                write_frames(False, 1)
            
            # Gap between letters (OFF)
            write_frames(False, 3)

    out.release()
    
    # Get absolute path to show you exactly where it is
    full_path = os.path.abspath(filename)
    print("-" * 30)
    print(f"SUCCESS!")
    print(f"Video saved at: {full_path}")
    print("-" * 30)

if __name__ == "__main__":
    # Fixed the function name here to match the definition above
    generate_morse_video()