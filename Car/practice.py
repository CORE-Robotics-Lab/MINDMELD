# Allows participant to practice driving around in environment
import requests
from CarEnv import CarEnv

car = CarEnv(calibration=False)
url = "http://127.0.0.1:5000/"

requests.post(url, json={"message": "run"})
url_data = "http://127.0.0.1:5000/data/"

car.client.enableApiControl(True)
car.reset(0)
