import yaml
import subprocess
from PyPDF2 import PdfMerger
import os

with open("window_size.yaml", "r") as f:
    config = yaml.safe_load(f)


while config["window_size"] <= 1.0:
    with open("window_size.yaml", "r") as f:
        config = yaml.safe_load(f)

    subprocess.run(["python3", "Classifier_all_data.py"])
    subprocess.run(["python3", "modell_test.py"])

    if config["window_size"] == 0.2:
        config["window_size"] = 0.3
    elif config["window_size"] == 0.7:
        config["window_size"] = 0.8
    else:
        config["window_size"] = config["window_size"] + 0.1
    if config["window_size"] <= 0.4:
        config["step_size"] = 0.05
    if config["window_size"] > 0.4:
        config["step_size"] = 0.1

    with open("window_size.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

pdf_folder = "./plots/"
merger = PdfMerger()

for i in range(1,11):  # 1 to 10
    file_path = os.path.join(pdf_folder, f"plot_{i}.pdf")
    merger.append(file_path)

merger.write("Window_sizes_plotted_1.pdf")
merger.close()
