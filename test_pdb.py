import pdb
import time

class TimedPdb(pdb.Pdb):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_time = time.time()

    def postcmd(self, stop, line):
        # Measure the time taken for the last command
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        print(f"[Execution Time] {elapsed_time:.6f} seconds")
        self.last_time = current_time
        return super().postcmd(stop, line)

def my_function():
    # Example function to debug
    x = 0
    for i in range(5):
        x += i
    return x

# Usage
pdb_instance = TimedPdb()
pdb_instance.set_trace()
my_function()