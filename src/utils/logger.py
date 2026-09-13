import csv
from pathlib import Path

class ControlLogger:
    def __init__(self, filepath=None):
        if filepath is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            filepath = project_root / 'results' / 'csv' / 'baseline_waypoint_controls.csv'
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['time', 'v', 'w', 'w_l', 'w_r'])

    def log(self, time, v, w, w_l, w_r):
        self.writer.writerow([f"{time:.4f}", f"{v:.4f}", f"{w:.4f}", f"{w_l:.4f}", f"{w_r:.4f}"])

    def close(self):
        self.file.close()