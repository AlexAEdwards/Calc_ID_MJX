import sys
from pathlib import Path

# Add the directory containing ProcessData to sys.path so we can import it
sys.path.append(str(Path.cwd()))
import ProcessData as pd
import numpy as np

# Let's inspect calculate_coupled_coordinates_automated
print(pd.calculate_coupled_coordinates_automated.__doc__)
