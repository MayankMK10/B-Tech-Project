import numpy as np
from scipy.io import wavfile

def generate_interactive_morse():
    # Full International Morse Code Dictionary
    MORSE_DICT = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
        '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
        '9': '----.', '0': '-----', ',': '--..--', '.': '.-.-.-', '?': '..--..'
    }
    
    # 1. Get user input
    text = input("Enter the sentence you want to convert to Morse audio: ")
    filename = "morse.wav"
    
    # 2. Settings
    fs = 44100       # Sample rate
    freq = 800       # Frequency of the beep in Hz
    dot_len = 0.1    # Duration of one unit (dot) in seconds
    audio = []

    # 3. Processing
    for char in text.upper():
        if char == ' ':
            # Standard Morse: 7 units of silence between words
            audio.extend(np.zeros(int(fs * dot_len * 7)))
            continue
            
        if char in MORSE_DICT:
            symbols = MORSE_DICT[char]
            for i, symbol in enumerate(symbols):
                # Dot = 1 unit, Dash = 3 units
                duration = dot_len if symbol == '.' else dot_len * 3
                
                t = np.linspace(0, duration, int(fs * duration), endpoint=False)
                beep = 0.5 * np.sin(2 * np.pi * freq * t)
                
                audio.extend(beep)
                
                # 1 unit of silence between dots/dashes of the same letter
                if i < len(symbols) - 1:
                    audio.extend(np.zeros(int(fs * dot_len)))
            
            # 3 units of silence between letters
            audio.extend(np.zeros(int(fs * dot_len * 3)))

    # 4. Save the file
    if len(audio) > 0:
        audio_data = (np.array(audio) * 32767).astype(np.int16)
        wavfile.write(filename, fs, audio_data)
        print(f"\nSuccess! '{filename}' created.")
        print(f"Total Audio Length: {len(audio)/fs:.2f} seconds")
    else:
        print("Error: No valid characters entered.")

if __name__ == "__main__":
    generate_interactive_morse()