import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import random
from smart_home_config import CONFIG
from occupancy_manager import OccupancyManager  # Adjust the import path as necessary

# Mock ConfigurationManager to return predefined CONFIG
mock_config_manager = MagicMock()
mock_config_manager.get_settings_config.side_effect = lambda key: CONFIG['settings'][key]
mock_config_manager.get_schedule_config.side_effect = lambda key: CONFIG['schedule'][key]
class TestOccupancyManager(unittest.TestCase):
    @patch('occupancy_manager.ConfigurationManager', return_value=mock_config_manager)
    def setUp(self, MockConfigManager):
        self.occupancy_manager = OccupancyManager()

    def test_initialization(self):
        """Test correct initialization of OccupancyManager with CONFIG settings."""
        self.assertEqual(self.occupancy_manager.current_time, CONFIG['settings']['start_of_simulation'])
        self.assertEqual(self.occupancy_manager.today, CONFIG['settings']['start_of_simulation'].strftime("%A"))

    @patch('occupancy_manager.random')
    def test_generate_schedule_for_person(self, mock_random):
        # Setup mock random to control variance and random event selection
        mock_random.randint.side_effect = [30, -30]  # Control variance application
        mock_random.random.return_value = 0.05  # Control random event trigger

        # Correctly pass the schedule for 'father' directly, not wrapped in another dict
        person_schedule = CONFIG['schedule']['weekly_schedule']['Monday']['father']
        self.occupancy_manager.random_event_chance = CONFIG['schedule']['random_event_chance']
        self.occupancy_manager.generate_schedule_for_person('father', person_schedule)  # Adjust method to accept person's name and schedule

        # Assertions follow...


        # Check if 'father' is in today's schedule
        self.assertIn('father', self.occupancy_manager.todays_schedule)

        # Test handling of a specific random event, e.g., 'sick-day'
        daily_schedule = self.occupancy_manager.todays_schedule['father']
        self.assertTrue('leave' in daily_schedule or 'at_home' in daily_schedule)

    def update_people_presence(self):
        for person, person_schedule in self.todays_schedule.items():
            if 'at_home' in person_schedule:
                self.people_presence[person] = person_schedule['at_home']
            else:
                leave_time = self.parse_time(person_schedule['leave'])
                return_time = self.parse_time(person_schedule['return'])
                self.people_presence[person] = not (leave_time <= self.current_time <= return_time)


        # Verify presence is updated correctly
        self.assertFalse(self.occupancy_manager.people_presence['father'])
        self.assertTrue(self.occupancy_manager.people_presence['mother'])
        self.assertFalse(self.occupancy_manager.people_presence['child'])

    # Additional test cases as needed...

if __name__ == '__main__':
    unittest.main()
