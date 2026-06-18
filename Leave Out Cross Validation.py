import yaml
import subprocess
from PyPDF2 import PdfMerger
import os
import numpy as np

fileNumber = 0
with open("file_to_skip.txt", "w") as f:
    f.write(str(fileNumber))

accuracies = []
while fileNumber <= 19:
    subprocess.run(["python3", "Classifier_all_data.py"])
    result = subprocess.run(["python3", "modell_test.py"],capture_output=True, text=True)
    accuracies.append(float(result.stdout.strip()))
    fileNumber = fileNumber +1
    with open("file_to_skip.txt", "w") as f:
        f.write(str(fileNumber))

print("The accuracies are:" ,accuracies)
print("Mean of the accuracies is: ", np.mean(accuracies))
print("standard deviation is: ", np.std(accuracies))

with open("LOCV_results", "w") as f:
    f.write(f"""The accuracies are: {accuracies}
        Mean of the accuracies is: {np.mean(accuracies)}
        Standard deviation is: {np.std(accuracies)}
        """)
'''
pdf_folder = "./plots_LOCV/"
merger = PdfMerger()

for i in range(1, 20):  # 1 to 19
    file_path = os.path.join(pdf_folder, f"plot_{i}.pdf")
    if os.path.exists(file_path):
        merger.append(file_path)

merger.write("LOCV.pdf")
merger.close()
'''