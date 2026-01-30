import Settings

class Logger:
    """Saves a log file (.txt)"""
    def __init__(self, save_path):
        self.file = open(save_path, mode = "w")

    def log(self, text, should_return = True):
        """Write the given text in the file. Automatically return to the next line, unless should_return is set to False."""
        self.file.write(text)
        if should_return:
            self.file.write("\n")

    def close(self):
        self.file.close()