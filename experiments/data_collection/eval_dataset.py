import os
import numpy as np
import pandas as pd
from pathlib import Path 

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
data_dir = PROJECT_ROOT / "data" / "intial_pos"
data_dir.mkdir(parents=True, exist_ok=True)

N=500
x_vals = np.random.uniform(-1.5,0 ,N)
y_vals = np.random.uniform(-2.5,2.5,N)
z_vals = np.zeros(N)

df = pd.DataFrame({'start_x': x_vals, 'start_y':y_vals, 'start_z':z_vals})
csv_path = data_dir / "start_positions_500.csv"
df.to_csv(csv_path, index=False)

print(f"Successfully generated {N} evaluation points at: {csv_path}")