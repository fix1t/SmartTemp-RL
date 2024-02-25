import csv
import random
from tools.logger_manager import Logger

class CSVLineReader:
    def __init__(self, file_path, start_from_random=False):
        self.file_path = file_path
        self.start_from_random = start_from_random
        self.reset(start_from_random=self.start_from_random)

    def reset(self, start_from_random=False):
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()

        self.file = open(self.file_path, 'r')
        self.csv_reader = csv.reader(self.file)

        if start_from_random:
            total_lines = self.get_number_of_lines()
            random_line = random.randint(0, total_lines - 1)
            for _ in range(random_line):
                next(self.csv_reader)

        self.generator = self._create_generator()

    def get_number_of_lines(self):
        with open(self.file_path, 'r') as f:
            return sum(1 for _ in f)

    def _create_generator(self):
        for row in self.csv_reader:
            yield row

    def get_next_line(self):
        try:
            row = next(self.generator)
        except StopIteration:
            self.file.close()
            self.reset(start_from_random=False)  # Reset if we've reached the end, starting from the beginning
            row = next(self.generator)

        if row and len(row) > 1:
            return row[0], self.get_float_line(row[1])

        Logger.error(f"Unexpected row format {row}")
        return row

    def get_float_line(self, line):
        try:
            return float(line)
        except ValueError:
            Logger.error(f"Warning: Cannot convert to float: {line}")
            return line

if __name__ == '__main__':
    # Example usage
    file_path = 'data/basel_10_years_hourly.csv'
    reader_random_start = CSVLineReader(file_path, start_from_random=True)

    print("Reading a few lines from a random starting point:")
    for _ in range(5):
        line = reader_random_start.get_next_line()
        print(line)

    reader_random_start.reset_to_beginning()
    print("\nResetting to the beginning and reading a few lines:")
    for _ in range(5):
        line = reader_random_start.get_next_line()
        print(line)
