import os
import csv


class HistoryCsvSaver:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_entry(self, epoch, metrics):
        all_metrics = {}
        all_metrics.update(metrics)

        row_dict = {'epoch': epoch}
        row_dict.update({k: self.scalar(v) for k, v in all_metrics.items()})

        field_names = list(row_dict.keys())

        if not os.path.exists(self.file_path):
            self.create(self.file_path, field_names)

        with open(self.file_path, 'a', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerow(row_dict)

    def scalar(self, t):
        return t.item() if hasattr(t, 'item') else t

    @classmethod
    def create(cls, path, field_names):
        with open(path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(field_names)
        return cls(path)
