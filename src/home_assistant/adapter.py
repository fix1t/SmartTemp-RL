import os
import requests
import json
from dotenv import load_dotenv

from mappers.action_mapper import ACTION_MAP
from mappers.sensor_mapper import SENSOR_MAP
class HomeAssistantAdapter:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.base_url = os.getenv('HOME_ASSISTANT_URL', 'http://localhost:8123')
        self.token = os.getenv('HOME_ASSISTANT_TOKEN')
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }

    def get_sensor_data(self, sensor_name):
        """Fetches the current state of a sensor based on sensor_name."""
        entity_id = SENSOR_MAP.get(sensor_name)
        if entity_id:
            url = f"{self.base_url}/api/states/{entity_id}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()['state']
        return None

    def execute_action(self, action_code):
        """Executes an action corresponding to the given action_code."""
        action, params = ACTION_MAP.get(action_code, (None, None))
        if action:
            url = f"{self.base_url}/api/services/homeassistant/{action}"
            response = requests.post(url, headers=self.headers, json=params)
            if response.status_code == 200:
                return True
        return False

    def collect_sensor_data(self):
        """Collects data from all sensors defined in SENSOR_MAP."""
        data = {}
        for sensor_name in SENSOR_MAP:
            data[sensor_name] = self.get_sensor_data(sensor_name)
        return data
