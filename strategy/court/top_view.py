import numpy as np
import cv2
from scipy import signal
import time
import matplotlib.pyplot as plt
from .plot import plot_speed_single, calculate_point_frequencies, plot_speed
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class TopViewProcessor:
    def __init__(self, players):
        self.players = players
        self.court = cv2.cvtColor(cv2.imread('strategy/court/court_reference.png'), cv2.COLOR_BGR2GRAY)
        self.court = cv2.cvtColor(self.court, cv2.COLOR_GRAY2BGR)
        self.inv_mats = []
        # for i in range(players):
        self.position = [[] for _ in range(players)]
        self.position1, self.position2 = [], []
        self.ball_position = []
        self.time_elapse = []

    def get_stats(self):
        return self.time_elapse, self.position, self.ball_position


    def visualize_bv(self, ball, humans):
        frame = self.court.copy()
        if ball[0] != -1:
            cv2.circle(frame, (int(ball[0]), int(ball[1])), 45, (0, 255, 0), -1)
        for human in humans:
            cv2.circle(frame, (int(human[0]), int(human[1])), 45, (0, 0, 255), -1)
        return frame

    def visualize_dummy(self):
        return self.court.copy()

    # def transform_player_location(self, matrix, locations):
    #     feet = np.array([(locations[0] + (locations[2] - locations[0]) / 2), locations[3]]).reshape((1, 1, 2))
    #     feet_court = cv2.perspectiveTransform(feet, matrix).reshape(-1)
    #     return feet_court

    def process(self, court_detector, players_boxes,ball_box, use_time, profile=False, vis_graph=True):
        self.time_elapse.append(use_time)
        # return self.court, self.court, self.court, self.court
        # img_h, img_w = inp_frame.shape[:2]
        """
        Calculate the feet position of both players using the inverse transformation of the court and the boxes
        of both players
        """
        if profile:
            init_time = time.time()
        frame = self.court.copy()
        if ball_box is not None:
            inv_mats = court_detector.game_warp_matrix
            ball_pos = np.array(ball_box).reshape(1,1,2)
            ball_court_pos = cv2.perspectiveTransform(ball_pos, inv_mats[-1]).reshape(-1)
            self.ball_position.append(ball_court_pos)
            frame = cv2.circle(frame,(int(self.ball_position[-1][0]),int(self.ball_position[-1][-1])), 45, (0, 255, 0), -1)

        if players_boxes is not None:
            if len(players_boxes) > self.players:
                players_boxes = players_boxes[:self.players]
            # assert len(players_boxes) == self.players, "The number of players is not correct!"
            inv_mats = court_detector.game_warp_matrix
            for idx, box in enumerate(players_boxes):
                box = box if isinstance(box, list) else box.item()
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
                feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[-1]).reshape(-1)
                self.position[idx].append(feet_court_pos)
                frame = cv2.circle(frame,(int(self.position[idx][-1][0]),int(self.position[idx][-1][-1])), 45, (0, 0, 255), -1)

            # self.inv_mats
            # positions_1 = []
            # positions_2 = []
            # Bottom player feet locations
            # for i, box in enumerate(player_1_boxes):
            # feet_pos = np.array([(player1_box[0] + (player1_box[2] - player1_box[0]) / 2).item(), player1_box[3].item()]).reshape((1, 1, 2))
            # feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[-1]).reshape(-1)
              # self.position1.append(feet_court_pos)
            # mask = []
            # # Top player feet locations
            # # for i, box in enumerate(player_2_boxes):
            # feet_pos = np.array([(player2_box[0] + (player2_box[2] - player2_box[0]) / 2), player2_box[3]])\
            #     .reshape((1, 1, 2))
            # feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[-1]).reshape(-1)
            # self.position2.append(feet_court_pos)
            # mask.append(True)

            # Smooth both feet locations




                # positions_1 = np.array(self.position1)
                # smoothed_1 = np.zeros_like(positions_1)
                # smoothed_1[:,0] = signal.savgol_filter(positions_1[:,0], window, 2)
                # smoothed_1[:,1] = signal.savgol_filter(positions_1[:,1], window, 2)
                # positions_2 = np.array(self.position2)
                # smoothed_2 = np.zeros_like(positions_2)
                # smoothed_2[:,0] = signal.savgol_filter(positions_2[:,0], window, 2)
                # smoothed_2[:,1] = signal.savgol_filter(positions_2[:,1], window, 2)

                # frame = cv2.circle(self.court.copy(), (int(smoothed_1[-1][0]), int(smoothed_1[-1][1])), 45, (0, 0, 255), -1)
                # frame = cv2.circle(frame, (int(smoothed_2[-1][0]), int(smoothed_2[-1][1])), 45, (0, 0, 255), -1)
        if not vis_graph:
            white_img = np.zeros((500, 400, 3), np.uint8)
            return frame, white_img, white_img, white_img
        if profile:
            print("Time taken for processing: ", time.time() - init_time)
            init_time = time.time()
        movement = self.vis_movement()
        if profile:
            print("Time taken for movement: ", time.time() - init_time)
            init_time = time.time()

        speed = self.vis_speed(use_time)
        if profile:
            print("Time taken for speed: ", time.time() - init_time)
            init_time = time.time()
        try:
            hm = self.vis_heatmap()
        except:
            hm = frame
        if profile:
            print("Time taken for heatmap: ", time.time() - init_time)
        return frame, movement, speed, hm

    def process_single(self, court_detector, players_boxes, use_time):
        # img_h, img_w = inp_frame.shape[:2]
        """
        Calculate the feet position of both players using the inverse transformation of the court and the boxes
        of both players
        """
        frame = self.court.copy()
        if players_boxes is not None:
            if len(players_boxes) == 2:
                player1_box, player2_box = players_boxes[0], players_boxes[1]
            else:
                raise ValueError("The number of players is not 2!")

            inv_mats = court_detector.game_warp_matrix
            # self.inv_mats
            # positions_1 = []
            # positions_2 = []
            # Bottom player feet locations
            # for i, box in enumerate(player_1_boxes):
            feet_pos = np.array([(player1_box[0] + (player1_box[2] - player1_box[0]) / 2).item(), player1_box[3].item()]).reshape((1, 1, 2))
            feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[-1]).reshape(-1)
            self.position1.append(feet_court_pos)
            mask = []
            # Top player feet locations
            # for i, box in enumerate(player_2_boxes):
            feet_pos = np.array([(player2_box[0] + (player2_box[2] - player2_box[0]) / 2), player2_box[3]])\
                .reshape((1, 1, 2))
            feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[-1]).reshape(-1)
            self.position2.append(feet_court_pos)
            mask.append(True)

            # Smooth both feet locations
            window = 7 if len(self.position1) > 7 else len(self.position1)
            if window >= 7:
                positions_1 = np.array(self.position1)
                smoothed_1 = np.zeros_like(positions_1)
                smoothed_1[:,0] = signal.savgol_filter(positions_1[:,0], window, 2)
                smoothed_1[:,1] = signal.savgol_filter(positions_1[:,1], window, 2)
                positions_2 = np.array(self.position2)
                smoothed_2 = np.zeros_like(positions_2)
                smoothed_2[:,0] = signal.savgol_filter(positions_2[:,0], window, 2)
                smoothed_2[:,1] = signal.savgol_filter(positions_2[:,1], window, 2)

                # smoothed_2[not mask, :] = [None, None]
                frame = cv2.circle(self.court.copy(), (int(smoothed_1[-1][0]), int(smoothed_1[-1][1])), 45, (0, 0, 255), -1)
                frame = cv2.circle(frame, (int(smoothed_2[-1][0]), int(smoothed_2[-1][1])), 45, (0, 0, 255), -1)
        # width = frame.shape[1] // 7
        # resized = imutils.resize(frame, height=int(img_h/2))
        # cv2.imshow("mipmap", resized)
        movement = self.vis_movement()
        # movement = imutils.resize(movement, height=int(img_h/2))
        speed = self.vis_speed(use_time)
        # speed = imutils.resize(speed, height=int(img_h/2))
        hm = self.vis_heatmap()
        # hm = imutils.resize(hm, height=int(img_h/2))
        # merged_img = np.concatenate((np.concatenate((resized, movement), axis=0),
        #                              np.concatenate((speed, hm), axis=0)), axis=1)
        # merged_img = imutils.resize(merged_img, height=img_h)
        # cv2.imshow("merged_result", merged_img)
        return frame, movement, speed, hm


    def get_player_location(self):
        # for i in range(self.players):
        if len(self.position[0]) == 0:
            return [[0, 0], [0, 0]]
        return [self.position[0][-1].tolist(), self.position[1][-1].tolist()]

    def get_ball_location(self):
        if len(self.ball_position) == 0:
            return[[0,0]]
        return [self.ball_position[-1].tolist()]


    def save(self, use_time=-1):
        self.vis_movement("tmp/movement_tmp.png")
        # movement = imutils.resize(movement, height=int(img_h/2))
        self.vis_speed(use_time, "tmp/speed_tmp.png")
        # speed = imutils.resize(speed, height=int(img_h/2))
        self.vis_heatmap("tmp/heatmap_tmp.png")

    # def vis(self, player1_pos, player2_pos):
    #     plt.plot(player1_pos)

    def draw_movement_line(self, ls, t, frame):
        for i in range(t):
            cv2.line(frame, (int(ls[-i][0]), int(ls[-i][1])), (int(ls[-i - 1][0]), int(ls[-i - 1][1])),
                     (0, 255, 0), 30)


    def vis_movement(self, save=""):
        max_movement_time = 5
        frame = self.court.copy()
        for pos in self.position:
            lines_num = min(max_movement_time, len(pos)-1)
            self.draw_movement_line(pos, lines_num, frame)
        if save:
            cv2.imwrite(save, frame)
        return frame
        # width = frame.shape[1] // 7
        # resized = imutils.resize(frame, width=width)
        # cv2.imshow("movement", resized)

    def vis_speed(self, use_time, save=""):
        max_speed_time = 20
        image = plot_speed(self.position, max_speed_time, use_time)
        if save:
            cv2.imwrite(save, image)
        return image
        img = cv2.imread("tmp/speed_tmp.png")
        return img
        cv2.imshow("speed", img)

    def vis_heatmap(self, save=""):
        # try:
        plt.clf()
        fig, ax = plt.subplots()
        concat_pos = np.concatenate(self.position, axis=0)
        # concat_pos = np.concatenate((self.position1, self.position2), axis=0)
        fre_map = calculate_point_frequencies(self.court.shape[1], self.court.shape[0], concat_pos, 6, 6)
        plt.imshow(np.array(fre_map))
        fig.canvas.draw()
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2RGB)
        if save:
            cv2.imwrite(save, img)
        return img
        # except:
        #     pass
        # return cv2.imread("tmp/heatmap_tmp.png")
        # cv2.imshow("heatmap", cv2.imread("heatmap_tmp.png"))

    def vis_movement_single(self):
        max_movement_time = 5
        frame = self.court.copy()
        lines_num1 = min(max_movement_time, len(self.position1)-1)
        lines_num2 = min(max_movement_time, len(self.position2)-1)
        self.draw_movement_line(self.position1, lines_num1, frame)
        self.draw_movement_line(self.position2, lines_num2, frame)
        return frame
        # width = frame.shape[1] // 7
        # resized = imutils.resize(frame, width=width)
        # cv2.imshow("movement", resized)

    def vis_speed_single(self, use_time):
        max_speed_time = 20
        plot_speed_single(self.position1, self.position2, max_speed_time, use_time)
        img = cv2.imread("tmp/speed_tmp.png")
        return img
        cv2.imshow("speed", img)

    def vis_heatmap_single(self):
        try:
            plt.clf()
            concat_pos = np.concatenate((self.position1, self.position2), axis=0)
            fre_map = calculate_point_frequencies(self.court.shape[1], self.court.shape[0], concat_pos, 6, 6)
            plt.imshow(np.array(fre_map))
            plt.savefig("tmp/heatmap_tmp.png")
            plt.clf()
        except:
            pass
        return cv2.imread("tmp/heatmap_tmp.png")
        cv2.imshow("heatmap", cv2.imread("heatmap_tmp.png"))

