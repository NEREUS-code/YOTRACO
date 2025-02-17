import json
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import openpyxl

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
        elif file_format == "xlsx": # Supports Excel
            wb = openpyxl.workbook()
            ws=wb.active
            ws.append(["Class" , "II Count", "OUT Count"])
            for key in data["in_counts"]:
                ws.append([key,data["in_counts"].get(key,0),data["out_counts"].get(key,0)])
            wb.save(filename)
        elif file_format == "txt": # Supports TXT
            with open(filename,"w") as f:
                f.write("Class\tIN Count\tOUT Count")
                for key in data["in_counts"]:
                    f.write(f"{key}\t{data['in_counts'].get(key,0)}\t{data["out_counts"].get(key,0)}\n")
        else:
            # TODO : support other format
            raise ValueError("Unsupported file format. Use 'json', 'csv','xlsx' or 'txt' .")
    
    def plot_bar(self):
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
    
    def plot_pie(self):
        """ 
        This function displays two pie charts:
        - One for incominf objects (IN Count)
        - One for outgoing objects (OUT Count) 
        """
        labels_in = list(self.class_counts_in.keys())
        sizes_in = list(self.class_counts_in.values())
        labels_out = list(self.class_counts_out.keys())
        sizes_out = list(self.class_counts_out.values())
        if not labels_in and not labels_out:
            raise ValueError("No data to display")
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        if labels_in:
            axs[0].pie(sizes_in, labels=labels_in, autopct="%1.1f%%",startangle=90,colors=plt.cm.Paired.colors)
            axs[0].set_title("Distribution of INCOMING objects")
        else:
            axs[0].set_title("No Incoming data") 

        if labels_out:
            axs[1].pie(sizes_out, labels=labels_out, autopct="%1.1f%%",startangle=90,colors=plt.cm.Paired.colors )
            axs[1].set_title("Distribution of Outgoing objects")
        else:
            axs[1].set_title("No Outgoing data ")
    
    def plot_scatter(self):
        """ 
        This function displays a scatter plot of incoming VS outcoming object counts.
        """
        labels_in = list(self.class_counts_in.keys())
        sizes_in = list(self.class_counts_in.values())
        labels_out = list(self.class_counts_out.keys())
        sizes_out = list(self.class_counts_out.values())
        if not labels_in and not labels_out:
            raise ValueError("No data to display")
        all_labels=list(set(labels_in).union(set(labels_out)))
        in_counts = [self.class_counts_in.get(label,0) for label in all_labels]
        out_count = [self.class_counts_out(label,0) for label in all_labels]
        plt.figure(figsize=(8, 6))
        plt.scatter(in_counts, out_count, color='green', label='Objects', alpha=0.7)
        plt.title("Scatter Plot of Incoming VS Outgoing Objects")
        plt.xlabel("Incoming Object Count")
        plt.ylabel("Outgoing Obejects Counts")
        plt.grid(True)
        plt.legend()
        plt.show()
        


