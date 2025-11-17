import sys
try:
    import sounddevice as sd
except Exception as e:
    print(f"ERROR_IMPORT:{e}")
    sys.exit(0)
try:
    devs = sd.query_devices()
    print('DEFAULT_DEVICE:', sd.default.device)
    for i, d in enumerate(devs):
        name = d.get('name') if isinstance(d, dict) else d
        print(i, name)
except Exception as e:
    print(f"ERROR_QUERY:{e}")
