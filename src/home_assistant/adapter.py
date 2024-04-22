import os
import requests
from dotenv import load_dotenv

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

    def get_sensor_data(self, entity_id):
        """Fetches the current state of a sensor."""
        url = f"{self.base_url}/api/states/{entity_id}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return None

    def post_data(self, entity_id, state):
        """Posts data to Home Assistant, for example, turning a light on or off."""
        url = f"{self.base_url}/api/states/{entity_id}"
        data = {'state': state}
        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return None
