import keyboard
import requests
from config import *

while True:
	keyboard.start_recording()
	keyboard.wait("s")
	requests.post(FLASK_URL, json={"message": "stop_record_demo"})
	print("sending stop")
	keyboard_events = keyboard.stop_recording()
