import os
import csv


folder_path = '/media/hkuit164/WD20EJRX/ESTRNN_dataset/labels'
output_csv_folder = "/media/hkuit164/WD20EJRX/ESTRNN_dataset/labels_out"
os.makedirs(output_csv_folder, exist_ok=True)
train_folder = f"{folder_path}/train"
val_folder = f"{folder_path}/val"


def write_csv(folder_path_txt, csv_folder):
    if folder_path_txt.split("/")[-1] == "train":
        csv_path = f"{csv_folder}/train0.csv"
    else:
        csv_path = f"{csv_folder}/val0.csv"

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        for file_name in os.listdir(folder_path_txt):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path_txt, file_name)
                write_line = []
                with open(file_path, 'r', encoding='utf-8') as txtfile:
                    for line in txtfile:
                        class_num = line.split(" ").pop(0)
                        write_line.append(line.split(" ") + [class_num] + [os.path.basename(file_path)])
                        new_list = [item for item in write_line[0] if item != '\n']
                        csv_writer.writerow(new_list)


def delete_rows(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            del row[:5]
            new_row = [cell for index, cell in enumerate(row) if (index + 1) % 3 != 0 and index < 54]
            writer.writerow(new_row)


write_csv(train_folder, output_csv_folder)
write_csv(val_folder, output_csv_folder)
delete_rows(f"{output_csv_folder}/train0.csv", f"{output_csv_folder}/train.csv")
delete_rows(f"{output_csv_folder}/val0.csv", f"{output_csv_folder}/val.csv")
