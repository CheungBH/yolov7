import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import csv

classes_list = ["forehand", "backhand", "overhead", "other"] # Add your desired classes here

input_csv_path = "/media/hkuit164/Backup/1.csv"
output_csv_path = "/media/hkuit164/Backup/2.csv"
folder_path = "/media/hkuit164/Backup/yolov7ps/yolov7/runs/detect/exp19/crops/person"  # Specify the crop image folder path directly
target_file = "classification_results.txt"  # Specify the target file name directly


class ImageLabellerGUI:
    def __init__(self, root):
        self.root = root
        self.image_files = []
        self.current_index = 0
        self.classification_results = {}
        self.classes = []

        self.create_widgets()
        self.load_images()
        self.read_results()

    def create_widgets(self):
        self.root.title("Image Classifier")

        # Create buttons frame at the top
        self.buttons_frame_top = tk.Frame(self.root)
        self.buttons_frame_top.pack(side=tk.TOP, pady=10)

        # Create back button
        self.back_button = tk.Button(self.buttons_frame_top, text="Back", command=self.previous_image)
        self.back_button.pack(side=tk.LEFT, padx=5)
        self.back_button.config(state=tk.DISABLED)

        # Create next button
        self.next_button = tk.Button(self.buttons_frame_top, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=5)

        # Create a frame to display the image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)

        # Create the text view
        self.text_view_frame = tk.Frame(self.root)
        self.text_view_frame.pack(pady=10)

        self.text_view_label = tk.Label(self.text_view_frame, text="Label:")
        self.text_view_label.pack(side=tk.LEFT)

        self.selected_image_label = tk.Label(self.text_view_frame, text="")
        self.selected_image_label.pack(side=tk.LEFT)

        # Create buttons frame at the bottom
        self.buttons_frame_bottom = tk.Frame(self.root)
        self.buttons_frame_bottom.pack(pady=10)

        # Create the counter label
        self.counter_label = tk.Label(self.buttons_frame_top, text="")
        self.counter_label.pack(side=tk.LEFT, padx=5)

        # Update the counter label
        self.update_counter_label()

        self.finish_button = tk.Button(self.root, text="Finish", command=self.show_warning)
        self.finish_button.pack(side=tk.TOP, pady=10)

    def update_counter_label(self):
        total_images = len(self.image_files)
        current_image_number = self.current_index + 1
        self.counter_label.config(text=f"{current_image_number}/{total_images}")

    def load_images(self):
        self.image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(self.image_files) == 0:
            messagebox.showerror("Error", "No image files found in the selected folder.")
            return
        self.classes = classes_list  # Add your desired classes here

        self.class_buttons = []
        for i, class_name in enumerate(self.classes):
            button = tk.Button(self.buttons_frame_bottom, text=class_name, command=lambda x=class_name: self.label_image(x))
            button.grid(row=0, column=i, padx=5)
            self.class_buttons.append(button)

        self.display_image()
        self.update_button_colors()

    def display_image(self):
        image_path = os.path.join(folder_path, self.image_files[self.current_index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to fit the GUI window
        max_height = 500
        max_width = 600
        image_height, image_width, _ = image.shape
        if image_height > max_height or image_width > max_width:
            scale = min(max_height / image_height, max_width / image_width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        self.photo = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        self.photo = Image.fromarray(self.photo)
        self.photo = ImageTk.PhotoImage(self.photo)

        self.image_label = tk.Label(self.image_frame, image=self.photo)
        self.image_label.pack()

        self.update_button_colors()
        self.update_counter_label()

    def read_results(self):
        if os.path.exists(target_file):
            with open(target_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    image_file, class_label = line.strip().split(":")
                    self.classification_results[image_file.strip()] = class_label.strip()
                self.update_text_view()

    def update_text_view(self):
        current_image = self.image_files[self.current_index]
        if current_image in self.classification_results:
            class_label = self.classification_results[current_image]
            self.selected_image_label.config(text=class_label)
        else:
            self.selected_image_label.config(text="")

    def finish_classification(self):
        self.save_results()
        messagebox.showinfo("Info", "Image labeling complete.")
        self.root.destroy()


    def label_image(self, class_name):
        current_image = self.image_files[self.current_index]
        self.classification_results[current_image] = class_name
        self.selected_image_label.config(text=class_name)
        self.update_button_colors()
        self.next_image()

    def next_image(self):
        self.current_index += 1
        if self.current_index < len(self.image_files):
            self.image_label.destroy()
            self.display_image()
            self.back_button.config(state=tk.NORMAL)
        else:
            self.save_results()
            messagebox.showinfo("Info", "Image labeling complete.")
            self.current_index -= 1

            # self.root.destroy()
        self.update_counter_label()

    def previous_image(self):
        self.current_index -= 1
        self.image_label.destroy()
        self.display_image()
        if self.current_index == 0:
            self.back_button.config(state=tk.DISABLED)
        else:
            current_image = self.image_files[self.current_index]
            if current_image in self.classification_results:
                current_class = self.classification_results[current_image]
                self.selected_image_label.config(text=current_class)
            else:
                self.selected_image_label.config(text="")
        self.update_counter_label()

    def update_button_colors(self):
        pass
        # for button in self.class_buttons:
        #     button.configure(bg="SystemButtonFace")  # Reset button color

        current_image = self.image_files[self.current_index]
        if current_image in self.classification_results:
            class_name = self.classification_results[current_image]
            self.selected_image_label.config(text=class_name, fg="red")  # Change text color to green
            for button in self.class_buttons:
                if button["text"] == class_name:
                    button.configure(bg="green")
        else:
            self.selected_image_label.config(text="")

    def show_warning(self):
        result = messagebox.askyesno("Warning", "Are you sure you want to finish labeling?")
        if result == tk.YES:
            self.finish_classification()

    def save_results(self):
        # Save the classification results to a text file
        with open(target_file, "w") as file:
            for image_file, class_label in self.classification_results.items():
                file.write(f"{image_file}: {class_label}\n")

        with open(input_csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
            last_column = [row[-1] for row in data]

        with open(output_csv_path, "w") as f:
            writer = csv.writer(f)
            for image_file, class_label in self.classification_results.items():
                for i, name in enumerate(last_column):
                    if name == image_file:
                        row = data[i]
                        row.append(class_label)
                        for j, cls in enumerate(classes_list):
                            if cls == class_label:
                                row.append(j)
                                writer.writerow(row)
                        break

        with open(output_csv_path, "r") as file:
            reader = csv.reader(file)
            rows = list(reader)

        for row in rows:
            last_column = row[-1]
            third_last_column = row[-3]
            row[-1] = third_last_column
            row[-3] = last_column

        with open(output_csv_path, "w") as file:
            writer = csv.writer(file)
            writer.writerows(rows)


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageLabellerGUI(root)
    root.mainloop()
