from gtts import gTTS
import os
import sys

# Get the text to convert from command-line arguments
text = sys.argv[1] if len(sys.argv) > 1 else "Default text for conversion."

# Language in which you want the audio
language = 'en'

# Convert the text to speech
speech = gTTS(text=text, lang=language, slow=False)

# Save the audio file
audio_file = r"output.mp3"
speech.save(audio_file)

# Play the audio (optional, works on most platforms)
os.system(f"start {audio_file}")  # Windows
# os.system(f"open {audio_file}")  # macOS
# os.system(f"xdg-open {audio_file}")  # Linux
