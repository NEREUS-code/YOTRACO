import json
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

class YotracoStats:
    def __init__(self):
        self.class_counts_in = defaultdict(int)
        self.class_counts_out = defaultdict(int)
    
    def get_counts(self):
        return {"in_counts" : dict(self.class_counts_in), "out_counts" : dict(self.class_counts_out)}
    
    def save_counts(self, filename, file_format = "json"):
        data = self.get_counts()
        if file_format == 'json':
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        elif file_format == "csv":
            with open(filename, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Class", "IN Count", "OUT Count"])
                for key in data["in_counts"].keys():
                    writer.writerow([key, data["in_counts"].get(key, 0), data["out_counts"].get(key,0)])
        else:
            # TODO : support other format
            raise ValueError("Unsupported file format. Use 'json' or 'csv' .")
    
    def plot_counts(self):
        labels = list(set(self.class_counts_in.keys()).union(set(self.class_counts_out.keys())))
        in_counts = [self.class_counts_in.get(label, 0) for label in labels]
        out_counts = [self.class_counts_out.get(label, 0) for label in labels]
        x = range(len(labels))
        plt.figure(figsize=(10, 5))
        plt.bar(x, in_counts, width=0.4, label="IN", color="green" , align="center")
        plt.bar(x, out_counts, width=0.4, label="OUT", color="red", align="edge")
        plt.xticks(x, labels, rotation=45)
        plt.ylabel("Count")
        plt.title("Object Count Tracking")
        plt.legend()
        plt.show()
        


