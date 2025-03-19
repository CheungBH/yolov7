import json
import csv

def json_to_csv(json_file_path, csv_file_path):
    # Read the JSON data from the file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    csv_content = []
    title = ["frame",] + list(data["0"].keys())
    csv_content.append(title)
    # Check if the JSON data is a list of dictionaries
    for key, value in data.items():
        row = [key,] + list(value.values())
        csv_content.append(row)

    # Write the CSV data to a file
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_content)

if __name__ == '__main__':
    json_file_path = '/Volumes/ASSETS/Tennis/datasets/raw_videos/general/MSc2023/Kang Hong/20231011_kh_yt_8_filter.json'
    csv_file_path = 'data.csv'
    json_to_csv(json_file_path, csv_file_path)
