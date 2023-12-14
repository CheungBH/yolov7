import os

detect_mode = "ML"  # ML or Temp

src_folder = "/media/hkuit164/Backup/xjl/20231207_kpsVideo/raw_video_all"
exp_name = "ml_train_4_knn_nobox"

sub_folders = os.listdir(src_folder)
sub_folders_path = [os.path.join(src_folder, sub_folder) for sub_folder in os.listdir(src_folder)]

for sub_folder, sub_folder_path in zip(sub_folders, sub_folders_path):
    if detect_mode == "ML":
        cmd = "python detect_with_ML.py --kpt-label --source {} --project runs/detect/{} --name {}".format(sub_folder_path, exp_name, sub_folder)
    else:  # Temp
        cmd = "python detect_with_temporal_kps.py --kpt-label --source {} --project runs/detect/{} --name {}".format(sub_folder_path, exp_name, sub_folder)
        print(cmd)
        os.system(cmd)
