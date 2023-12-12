
import os


src_folder = "/media/hkuit164/Backup/xjl/20231207_kpsVideo/raw_video"
json_folder = "json_test"

actions = os.listdir(src_folder)
actions_path = [os.path.join(src_folder, action) for action in os.listdir(src_folder)]
for action, action_path in zip(actions, actions_path):
    videos_name = os.listdir(action_path)
    os.makedirs(os.path.join(json_folder, action), exist_ok=True)

    for video_name in videos_name:
        video_path = os.path.join(src_folder, action, video_name)
        target_json = os.path.join(json_folder, action, video_name.split(".")[0] + ".json")
        # os.makedirs()
        cmd = "python gen_temporal.py --source {} --kpt-label --nosave --json-path {}".format(video_path, target_json)
        print(cmd)
        os.system(cmd)
