import sys
import runpy
try:
    import sounddevice as sd
except Exception as e:
    print(f"sounddevice import error: {e}")
    sys.exit(1)
# set input device index 1 (your chosen microphone) and keep output as current default (3)
sd.default.device = (1, 3)
print(f"Set sounddevice default device to: {sd.default.device}")
# Run the assistant script in simulate mode to avoid requiring TensorFlow
sys.argv = ['realtime_assistant.py', '--simulate']
runpy.run_path('Speech Commands Dataset\\realtime_assistant.py', run_name='__main__')
