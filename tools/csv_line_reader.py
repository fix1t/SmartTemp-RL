import csv
from tools.logger_manager import Logger

class CSVLineReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self._reset()

    def _reset(self):
        # Close the file if it's already open
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
        # Open the file, setup the CSV reader and the generator
        self.file = open(self.file_path, 'r')
        self.csv_reader = csv.reader(self.file)
        self.generator = self._create_generator()

    def _create_generator(self):
        for row in self.csv_reader:
            yield row

    def get_next_line(self):
        try:
            row = next(self.generator)
            # Convert the element at index 1 to float, if it exists and is convertible
            if row and len(row) > 1:
                try:
                    row[1] = float(row[1])
                except ValueError:
                    Logger.error(f"Warning: Cannot convert to float: {row[1]}")
            return row
        except StopIteration:
            self.file.close()
            return None

    """
    Resets the reading cursor back to the beginning of the file.
    """
    def reset_to_beginning(self):
        self._reset()
