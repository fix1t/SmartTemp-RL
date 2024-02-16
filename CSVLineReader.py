import csv

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
            return next(self.generator)
        except StopIteration:
            self.file.close()
            #TODO: handle EOF
            return None

    def reset_to_beginning(self):
        # Resets the reading cursor back to the beginning of the file.
        self._reset()

