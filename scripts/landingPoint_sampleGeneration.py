
import os
#input_file = "/Users/chris/label_studio_concise/coordinates.txt"


def process_single_input(file, interval=1, adjacent=4):
    with open(file, 'r', encoding='utf-8') as f:
        content = [line.split(",") for line in f.readlines()]
    samples, labels = [], []
    # label0, label1 = [], []
    duration = adjacent * 2 + 1
    for i in range(len(content)-duration):
        # last_idx = i + adjacent*2 + 1
        if int(content[i+duration-1][-1][:-1]) == -1:
            continue
        sample = []
        for j in range(duration):
            sample.append(content[i+j][0])
            sample.append(content[i+j][1])
        samples.append(sample)
        labels.append(int(content[i+duration-1][-1][:-1]))
        
    return samples, labels


file_folder = "/Volumes/ASSETS/tmp/4.4/landing_data_MSc2023_serve/txt"
files = [os.path.join(file_folder, file) for file in os.listdir(file_folder) if file.endswith(".txt") and not file.startswith(".")]
samples, labels = [], []
for file in files:
    print("Processing file: {}".format(file))
    s, l = process_single_input(file)
    samples += s
    labels += l

csv_file = "/Volumes/ASSETS/tmp/4.4/landing_data_MSc2023_serve/train.csv"
with open(csv_file, "w") as f:
    for sample, label in zip(samples, labels):
        sample_str = ",".join(sample) + ",{}\n".format(label)
        f.write(sample_str)



