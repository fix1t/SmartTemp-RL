import csv

class CSVLineReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(self.file_path, 'r')
        self.csv_reader = csv.reader(self.file)
        self.generator = self._create_generator()

    def _create_generator(self):
        for row in self.csv_reader:
            yield row

    def get_next_line(self):
        try:
            return next(self.generator)
        except StopIteration:
            # Optionally, close the file once all lines are read
            self.file.close()
            # Return None or raise an exception if you prefer when the end of the file is reached
            return None
