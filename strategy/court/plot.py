import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2


class Plotter:
    def __init__(self):
        pass

    def visualize(self):
        pass


def cal_dis(coor1, coor2):
    out = np.square(coor1[0] - coor2[0]) + np.square(coor1[1] - coor2[1])
    return np.sqrt(out)


def calculate_movement(ls):
    move = []
    for idx in range(len(ls)-1):
        move.append(cal_dis(ls[idx], ls[idx+1]))
    return move


def plot_speed(players, max_time=20, use_time=0):
    plt.clf()
    fig, ax = plt.subplots()# = Figure(figsize=(5, 4), dpi=100)
    # canvas = FigureCanvasAgg(fig)
    if len(players[0]) < 2:
        array = np.array([[0,0] for _ in range(len(players))])
    else:
        array = [calculate_movement(player) for player in players]

    # Determine the number of elements to plot
    num_elements = min(max_time, len(players[0]))
    valid_array = [[(x / use_time) / 100 for x in array[-num_elements:]] for array in array]
    # Plot the curves
    for i in range(len(players)):
        plt.plot(valid_array[i], label=f'Player {i+1}')

    # Add labels and title
    # ax.xlabel('Time')
    # ax.ylabel('Speed (m/s)')
    # ax.title('Speed of players')

    ax.legend()
    fig.canvas.draw()

    img_plot = np.array(fig.canvas.renderer.buffer_rgba())

    # convert to a NumPy array
    # X = np.asarray(buf)
    # canvas = fig.canvas
    return cv2.cvtColor(img_plot, cv2.COLOR_RGBA2RGB)

    # plt.savefig("tmp/speed_tmp.png")
    # plt.clf()


def plot_speed_single(player1, player2, max_time=20, use_time=0):
    if len(player1) < 2 or len(player2) < 2:
        array1 = np.array([[0,0], [0,0]])
        array2 = np.array([[0,0], [0,0]])
    else:
        array1 = calculate_movement(player1)
        array2 = calculate_movement(player2)

    # Determine the number of elements to plot
    num_elements = min(max_time, len(array1), len(array2))

    # valid_array1 = array1[-num_elements:]
    # valid_array2 = array2[-num_elements:]
    valid_array1 = [(x / use_time) / 100 for x in array1[-num_elements:]]
    valid_array2 = [(x / use_time) / 100 for x in array2[-num_elements:]]
    # x_coord = [idx/10 for idx in range(20)]
    # Plot the curves
    plt.plot(valid_array1, label='Player 1')
    plt.plot(valid_array2, label='Player 2')

    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    plt.title('Speed of players')

    plt.legend()

    plt.savefig("tmp/speed_tmp.png")
    plt.clf()


def calculate_point_frequencies(area_width, area_height, points, grid_rows, grid_cols):
    grid_width = area_width // grid_cols
    grid_height = area_height // grid_rows
    frequencies = [[0] * grid_cols for _ in range(grid_rows)]

    for point in points:
        x, y = point
        grid_x = int(x // grid_width)
        grid_y = int(y // grid_height)
        frequencies[grid_y][grid_x] += 1

    return frequencies


if __name__ == '__main__':
    area_width = 12
    area_height = 16
    points = [(7, 1), (4, 1)]
    grid_rows = 2
    grid_cols = 4

    frequencies = calculate_point_frequencies(area_width, area_height, points, grid_rows, grid_cols)
    for row in frequencies:
        print(row)
