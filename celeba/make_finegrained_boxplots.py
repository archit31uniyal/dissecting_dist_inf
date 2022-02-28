import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils import flash_utils, get_n_effective, bound
from data_utils import SUPPORTED_PROPERTIES
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        default="Male",
                        help='name for subfolder to save/load data from')
    parser.add_argument('--mode', choices=["meta", "threshold"],
                        default="meta")
    args = parser.parse_args()
    flash_utils(args)

    # Set font size
    plt.rcParams.update({'font.size': 6})
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('axes', labelsize=10)

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    targets = ["0.0", "0.1", "0.2", "0.3", "0.4",
               "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

    fill_data = np.zeros((len(targets), len(targets)))
    mask = np.ones((len(targets), len(targets)), dtype=bool)
    annot_data = [[None] * len(targets) for _ in range(len(targets))]
    if args.filter == "Male":
        # Combined
        raw_data_meta = [
            [
                [58.3, 56.8, 57.6, 58.35, 53.95],
                [63.35, 65.8, 66.05, 61.2, 65.35],
                [76.7, 69.8, 76.25, 67.5, 74.0],
                [77.85, 78.2, 82.3, 87.15, 85.2],
                [93.2, 89.65, 93.25, 92.0, 91.65],
                [95.25, 90.45, 88.45, 91.5, 93.3],
                [95.15, 95.75, 94.35, 94.25, 95.2],
                [87.4, 85.25, 65.25, 90.4, 89.85],
                [95.45, 87.15, 93.45, 96.7, 86.45],
                [75.25, 61.55, 62.1, 67.1, 76.15]
            ],
            [
                [52.55, 55.35, 52.0, 51.55, 53.95],
                [61.7, 61.85, 60.2, 63.5, 60.6],
                [68.6, 74.75, 66.15, 70.9, 72.9],
                [68.55, 81.35, 77.5, 74.85, 80.65],
                [74.45, 85.95, 86.55, 83.5, 86.25],
                [88.6, 91.2, 91.15, 89.05, 89.2],
                [80.8, 69.15, 74.25, 70.6, 65.05],
                [80.7, 86.7, 83.55, 86.25, 89.55],
                [80.4, 67.95, 61.0, 81.9, 68.85]
            ],
            [
                [52.7, 52.85, 51.6, 52.85, 52.75],
                [57.8, 58.85, 57.75, 58.45, 58.0],
                [67.5, 61.7, 65.9, 59.5, 59.3],
                [68.15, 74.1, 77.4, 73.75, 73.8],
                [80.6, 69.15, 79.85, 76.95, 78.35],
                [81.2, 77.4, 77.75, 74.5, 81.85],
                [83.0, 82.7, 80.4, 86.75, 84.4],
                [76.95, 67.9, 67.9, 74.25, 67.2]
            ],
            [
                [54.9, 54.85, 54.15, 55.5, 51.75],
                [60.95, 57.8, 60.45, 60.0, 56.3],
                [70.55, 74.5, 71.95, 71.85, 64.0],
                [79.7, 80.45, 72.25, 72.8, 62.85],
                [75.8, 76.95, 62.05, 62.75, 60.95],
                [80.4, 80.9, 72.75, 86.05, 85.25],
                [67.2, 65.65, 65.5, 67.85, 71.45]
            ],
            [
                [53.0, 52.95, 54.0, 53.9, 51.55],
                [62.35, 60.8, 60.8, 61.5, 55.7],
                [67.2, 69.65, 67.95, 68.3, 67.75],
                [51.95, 74.95, 51.65, 60.65, 52.0],
                [76.55, 72.25, 66.25, 65.8, 64.85],
                [59.85, 65.25, 60.35, 62.2, 59.65]
            ],
            [
                [52.5, 53.35, 54.75, 53.35, 51.5],
                [59.7, 55.5, 61.35, 60.8, 62.2],
                [51.85, 62.9, 52.1, 60.25, 51.85],
                [70.4, 74.2, 73.1, 70.5, 69.1],
                [67.45, 66.2, 60.1, 69.05, 65.4]
            ],
            [
                [52.2, 50.05, 52.3, 51.85, 53.6],
                [48.1, 49.75, 48.15, 51.3, 51.4],
                [60.05, 63.3, 65.45, 65.9, 61.65],
                [58.85, 59.25, 62.4, 58.0, 59.25]
            ],
            [
                [49.55, 49.3, 49.55, 49.95, 49.5],
                [55.95, 54.5, 57.9, 58.8, 56.55],
                [55.0, 52.55, 58.2, 58.25, 57.9]
            ],
            [
                [48.45, 47.2, 47.7, 48.8, 48.5],
                [52.8, 54.85, 54.5, 56.5, 54.3]
            ],
            [
                [50.55, 51.4, 50.9, 53.6, 54.0]
            ],
        ]

        raw_data_regress = [
            [
                [54.85, ],
                [62.45, ],
                [71.7, ],
                [82.15, ],
                [90.45, ],
                [95.3, ],
                [97.55, ],
                [97.95, ],
                [97.55, ],
                [94.6, ],
            ],
            [
                [59.95, ],
                [67.25, ],
                [78.05, ],
                [86.9, ],
                [92.1, ],
                [95.2, ],
                [94.75, ],
                [93.8, ],
                [87.05, ],
            ],
            [
                [59.95, ],
                [69.85, ],
                [79.2, ],
                [86.45, ],
                [89.4, ],
                [87.65, ],
                [86.2, ],
                [77.45, ],
            ],
            [
                [63.7, ],
                [72.35, ],
                [78.15, ],
                [80.4, ],
                [77.85, ],
                [75.2, ],
                [67.85, ],
            ],
            [
                [61.05, ],
                [66.65, ],
                [68.7, ],
                [68.05, ],
                [65.75, ],
                [59.25, ],
            ],
            [
                [56.9, ],
                [59.45, ],
                [59.65, ],
                [57.5, ],
                [54.15, ],
            ],
            [
                [52.6, ],
                [54.3, ],
                [53.3, ],
                [51.65, ],
            ],
            [
                [51.2, ],
                [50.85, ],
                [50.65, ],
            ],
            [
                [50.25, ],
                [50.1, ],
            ],
            [
                [50, ],
            ]
        ]
        # With Conv
        # raw_data_meta = [
        #     [
        #         [55.9, 56.25, 55.1, 57.85, 56.8],
        #         [67.85, 66.1, 63.0, 63.75, 64.85],
        #         [72.2, 73.05, 76.85, 76.8, 68.2],
        #         [82.95, 85.75, 79.7, 83.2, 80.4],
        #         [83.2, 85.5, 88.9, 82.1, 80.9],
        #         [88.15, 91.95, 92.65, 89.75, 93.95],
        #         [94.4, 94.75, 96.8, 97.4, 93.8],
        #         [89.2, 80.3, 74.6, 83.5, 66.4],
        #         [96.2, 89.95, 93.3, 94.7, 92.75],
        #         [78.15, 77.5, 75.55, 71.65, 72.7]
        #     ],
        #     [
        #         [51.85, 54.1, 55.6, 54.0, 51.65],
        #         [62.35, 58.9, 60.15, 61.4, 62.7],
        #         [67.4, 72.6, 70.9, 70.7, 68.55],
        #         [78.4, 78.55, 76.9, 83.35, 76.5],
        #         [88.65, 89.05, 77.55, 86.65, 88.65],
        #         [72.45, 86.3, 86.0, 87.15, 92.35],
        #         [61.0, 78.8, 85.3, 75.5, 83.7],
        #         [88.9, 87.6, 79.6, 86.85, 82.45],
        #         [75.15, 68.1, 70.3, 68.95, 71.65]
        #     ],
        #     [
        #         [53.1, 52.9, 52.45, 50.2, 52.2],
        #         [58.4, 58.05, 54.15, 54.15, 60.45],
        #         [66.4, 69.5, 61.85, 62.6, 67.95],
        #         [74.85, 75.9, 68.4, 73.1, 71.0],
        #         [83.6, 79.3, 71.05, 68.25, 80.75],
        #         [83.55, 75.5, 69.1, 75.45, 78.9],
        #         [82.65, 77.2, 72.55, 74.75, 81.3],
        #         [68.0, 63.6, 79.2, 66.1, 59.0]
        #     ],
        #     [
        #         [55.45, 54.45, 53.45, 54.7, 54.55],
        #         [57.85, 63.5, 56.65, 57.85, 59.15],
        #         [66.8, 71.8, 63.5, 75.0, 74.85],
        #         [80.9, 69.9, 78.9, 74.55, 80.4],
        #         [56.45, 52.25, 56.95, 70.45, 61.0],
        #         [86.75, 65.95, 80.45, 81.45, 79.15],
        #         [67.95, 68.4, 61.95, 60.85, 58.65]
        #     ],
        #     [
        #         [51.05, 53.95, 53.5, 54.95, 53.65],
        #         [55.5, 59.5, 62.1, 58.5, 60.8],
        #         [71.05, 68.4, 70.65, 69.7, 67.45],
        #         [51.5, 55.65, 63.95, 54.5, 55.45],
        #         [61.25, 65.7, 78.25, 71.8, 75.95],
        #         [69.2, 71.35, 77.65, 61.4, 66.8]
        #     ],
        #     [
        #         [55.55, 55.2, 53.8, 53.35, 52.1],
        #         [60.15, 61.55, 60.0, 62.4, 62.55],
        #         [50.9, 50.25, 51.25, 53.35, 55.15],
        #         [63.4, 61.35, 66.6, 72.2, 69.15],
        #         [60.75, 57.7, 62.5, 61.0, 55.0]
        #     ],
        #     [
        #         [53.15, 53.45, 50.95, 54.15, 52.4],
        #         [49.55, 48.35, 49.25, 48.65, 48.5],
        #         [64.95, 65.75, 64.65, 60.55, 62.05],
        #         [61.95, 58.7, 62.35, 63.6, 62.0]
        #     ],
        #     [
        #         [49.85, 49.85, 49.65, 49.3, 50.0],
        #         [57.3, 59.0, 58.05, 57.4, 56.95],
        #         [58.7, 57.45, 56.25, 55.8, 56.6]
        #     ],
        #     [
        #         [48.2, 49.1, 48.45, 48.45, 46.9],
        #         [54.1, 52.8, 53.15, 51.35, 53.75]
        #     ],
        #     [
        #         [50.65, 51.7, 50.45, 50.95, 50.7]
        #     ]
        # ]

        raw_data_threshold = [
            [
                [50.68, 50.03, 51.24],
                [56.29, 55.24],  # ?], # Running
                [55.56, 52.84, 55.20],
                [56.25, 57.01, 61.31],
                [54.2, 56.9, 58.05],
                [59.09, 60.98, 61.03],
                [67.37, 61.94, 63.95],
                [56.09, 55.24],  # ?], # Running
                [76.65, 76.49, 75.68],
                [76.1, 76.4, 74.8]
            ],
            [
                [51.15, 48.9, 52.36],
                [50.36, 51.98, 52.74],
                [53.83, 51.43, 53.67],
                [50.48, 55.57, 53.66],
                [58.63, 59.97, 57.71],
                [66.79, 59.99, 64.71],
                [48.05, 50.1, 44.74],
                [74.50, 73.43, 71.79],
                [73.58, 75.59, 75.79]
            ],
            [
                [51.05, 51.65, 50.65],
                [51.63, 51.95, 50.5],
                [54.56, 55.12, 50.93],
                [59.67, 58.67, 59.82],
                [62.68, 64.83, 65.93],
                [48.8, 46.94, 49.2],
                [72.8, 72.24, ],  # ?], #Running
                [77.49, 74.34, 78.0]
            ],
            [
                [50.15, 50.92, 50.61],
                [52.09, 53.34, 52.44],
                [54.05, 56.76, 55.17],
                [59, 63.2, 62],
                [41.58, 44.49],  # ?], #Running
                [67.65, 66.89, 71.57],
                [73.71, 71.54, 73.91]
            ],
            [
                [51.14, 52.05, 50.97],
                [56.34, 52.89, 56.91],
                [61.34, 58.24, 61.24],
                [43.99, 43.84, 39.68],
                [71.91, 65.71, 67.5],
                [68.64, 74.05, 73.04]
            ],
            [
                [55.68, 55.41, 51.96],
                [60.68, 60.43, 60.18],
                [54.48, 55.09, 50.0],
                [66.26, 68.39, 67.63],
                [72.45, 72.4, 69.05]
            ],
            [
                [53.79, 55.99, 56.3],
                [45.09, 47.14, 46.09],
                [62.1, 65.05, 59.94],
                [66.43, 69.59, 67.19]
            ],
            [
                [48.4, 48.3, 48.7],
                [58.64, 55.63, 57.52],
                [59.23, 62.75, 56.66]
            ],
            [
                [49.4, 49.5, 49.9],
                [52.31, 51.43, 51.33],
            ],
            [
                [54.86, 54.51, 53.7]
            ]
        ]

        raw_data_loss = [
            [53.12, 50.75, 52.13, 51.44, 51.15, 52.92, 50.86, 50.55, 51.74, 50.4],
            [50.1, 52.01, 51.59, 50.53, 50.14, 50.6, 50.5, 52.63, 50.19],
            [51.6, 50.85, 50.36, 50.83, 50.19, 50.84, 49.95, 50.14],
            [50.06, 50.01, 54.65, 51.06, 48.75, 50.32, 49.64],
            [50.44, 56.9, 51.91, 50.1, 52.43, 50.19],
            [50.78, 57.26, 51.5, 46.62, 50.1],
            [52.19, 50.95, 51.53, 48.62],
            [49.15, 50.1, 49.85],
            [50.4, 49.15],
            [50.42]
        ]

    elif args.filter == "Young":
        raw_data_meta = [
            [
                [50.35, 51.1, 49.65, 53.3, 49.9],
                [50.4, 57.25, 49.75, 59.85, 55.25],
                [68.15, 56.05, 65.15, 63.5, 69.2],
                [61.45, 72.8, 75.9, 75.4, 74.85],
                [78.85, 50.1, 79.8, 79.2, 75.15],
                [84.6, 84.2, 83.85, 78.05, 79.9],
                [84.8, 49.9, 88.45, 89.2, 88.75],
                [84.25, 84.05, 90.65, 91.4, 88.1],
                [93.7, 89.6, 94.5, 93.4, 93.75],
                [94.5, 94.8, 95.3, 95.0, 95.9]
            ],
            [
                [50.95, 49.7, 49.15, 48.25, 47.85],
                [58.2, 50.85, 57.95, 57.25, 48.95],
                [50.25, 65.1, 50.8, 62.2, 63.65],
                [70.9, 68.15, 70.1, 70.15, 69.05],
                [75.1, 62.25, 71.8, 73.25, 76.2],
                [79.6, 80.1, 79.0, 80.45, 78.6],
                [84.4, 51.05, 80.45, 76.15, 84.95],
                [89.55, 89.75, 83.35, 84.95, 89.8],
                [92.0, 92.05, 92.35, 93.05, 92.05]
            ],
            [
                [51.7, 51.45, 48.5, 51.3, 49.8],
                [55.15, 54.0, 56.65, 58.65, 52.15],
                [60.05, 63.6, 63.55, 63.55, 63.8],
                [68.15, 66.05, 50.35, 49.0, 65.15],
                [74.25, 75.5, 51.0, 73.65, 73.7],
                [80.15, 79.2, 79.3, 80.05, 78.75],
                [83.9, 80.9, 77.9, 84.75, 83.7],
                [89.85, 88.7, 90.6, 90.55, 90.2],
            ],
            [
                [51.2, 51.1, 49.7, 51.0, 49.5],
                [54.15, 52.75, 53.0, 50.9, 56.7],
                [48.55, 59.4, 57.9, 57.2, 60.1],
                [67.4, 67.15, 53.55, 51.0, 66.6],
                [71.65, 71.95, 72.15, 58.5, 59.15],
                [74.0, 76.35, 79.5, 78.05, 79.95],
                [84.2, 85.05, 85.5, 49.6, 84.7],
            ],
            [
                [48.7, 50.7, 49.35, 51.1, 49.4],
                [53.85, 56.1, 53.9, 51.1, 54.9],
                [55.7, 59.55, 50.0, 60.15, 60.25],
                [65.3, 63.55, 65.05, 65.35, 67.35],
                [67.4, 73.65, 73.9, 50.0, 71.35],
                [80.45, 79.95, 67.65, 69.75, 80.05]
            ],
            [
                [49.45, 49.35, 49.55, 50.6, 51.8],
                [50.15, 52.1, 55.4, 54.75, 49.5],
                [61.65, 49.95, 54.55, 50.75, 61.15],
                [66.1, 66.3, 64.45, 62.55, 67.0],
                [73.85, 50.85, 74.95, 76.4, 74.05]
            ],
            [
                [52.1, 48.8, 51.4, 51.55, 50.0],
                [51.1, 52.1, 52.6, 51.8, 48.85],
                [55.2, 61.15, 62.45, 63.15, 58.15],
                [69.95, 69.8, 68.1, 50.0, 65.6]
            ],
            [
                [49.85, 50.4, 50.3, 49.55, 52.0],
                [54.0, 53.8, 49.75, 56.6, 53.55],
                [64.05, 66.2, 59.95, 59.6, 55.35]
            ],
            [
                [51.7, 51.5, 50.5, 51.25, 50.15],
                [50.0, 56.75, 58.35, 54.45, 50.85]
            ],
            [
                [49.85, 52.75, 52.45, 49.8, 51.3]
            ]
        ]
        # # OLD
        # raw_data_meta = [
        #     [
        #         [50.74, 54.47, 52.7],
        #         [60.06, 50.29, 49.71],
        #         [63.23, 63.5, 61.17],
        #         [66.17, 61.62, 68.26, 63.23, 63.49, 61.17],
        #         [80.05, 74.6, 79.2, 76.95, 78.15],
        #         [81.62, 81.62, 81.72],
        #         [84.55, 82.37, 80.6],
        #         [85.13, 86.66, 86.53],
        #         [83.31, 87.15, 81.32],
        #         [85.6, 87.26, 83.6]
        #     ],
        #     [
        #         [51.42, 50.95, 51.03],
        #         [55.64, 57.78, 50.68],
        #         [50.23, 63.84, 61.76],
        #         [71.05, 70.77, 68.85, 70.87, 71.59],
        #         [76.43, 74.58, 73.84],
        #         [75.31, 80.04, 77.81],
        #         [48.92, 80.57, 81.85],
        #         [84.08, 86.87, 82.28],
        #         [87.5, 86.73, 50.01]
        #     ],
        #     [
        #         [52.12, 50.01, 52.14],
        #         [56.75, 49.2, 49.86],
        #         [62.27, 61.41, 64.04, 60.55, 51.06],
        #         [64.93, 67.63, 60.48],
        #         [67.82, 73.86, 73.81],
        #         [74.8, 73.6, 74.96],
        #         [74.2, 76.55, 80.52],
        #         [82.94, 80.25, 80.22]
        #     ],
        #     [
        #         [49.35, 50.26, 49.19],
        #         [51.07, 54.46, 48.93, 53.03, 53.65],
        #         [58.72, 59.67, 59.40],
        #         [62.25, 65.91, 65.39],
        #         [66.39, 69.06, 68.6],
        #         [74.06, 71.77, 70.46],
        #         [75.78, 51.03, 76.46]
        #     ],
        #     [
        #         [52.95, 52.67, 51.64, 50.03, 49.44],
        #         [50.12, 54.47, 54.85],
        #         [57.56, 59.77, 59.67],
        #         [50.85, 50.86, 64.26],
        #         [70.73, 67.01, 67.98],
        #         [75.47, 69.2, 75.39]
        #     ],
        #     [
        #         [51.65, 51.24, 51.37, 51.27, 51.52],
        #         [54.24, 54.04, 53.32, 53.35, 54.37],
        #         [58.94, 51.11, 58.21, 48.89, 58.26],
        #         [64.08, 62.8, 64.86, 63.58, 64.15],
        #         [69.26, 69.82, 72.60, 71.3, 73.09]
        #     ],
        #     [
        #         [51.71, 50.17, 51.37],
        #         [53.24, 49.03, 53.01],
        #         [60.99, 50.94, 57.17],
        #         [65.4, 65.38, 66.14]
        #     ],
        #     [
        #         [51.38, 50.54, 50.93],
        #         [53.46, 55.68, 48.99],
        #         [62.86, 62.55, 62.45]
        #     ],
        #     [
        #         [52.05, 50.72, 50.91],
        #         [57.47, 57.55, 58.05]
        #     ],
        #     [
        #         [51.36, 51.67, 48.93]
        #     ]
        # ]
        raw_data_threshold = [
            [
                [49.77, 49.72, 50.28],
                [48.37, 54.76, 48.27],
                [50.25, 50.35, 50.25],
                [48.32, 50.3, 50.25],
                [50.27, 50.28, 50.28],
                [49.07, 49.02, 48.92],
                [50.03, 50.03, 50.41],
                [39.52, 49.52, 49.52],
                [76.06, 50.33, 50.23],
                [50.48, 88.89, 50.63]
            ],
            [
                [47.94, 52.1, 52.06],
                [49.98, 49.98, 49.98],
                [50.65, 49.75, 42.7],
                [49.95, 50, 49.95],
                [48.13, 52.07, 48.08],
                [49.15, 49.55, 49.65],
                [49.14, 49.19, 49.14],
                [58.83, 49.85, 67.85],
                [80.5, 75.35, 80.3]
            ],
            [
                [52.03, 52.03, 47.97],
                [52.06, 48.52, 52.0],
                [51.95, 48, 52.06],
                [49.79, 55.81, 50.16],
                [55.31, 51.28, 50.97],
                [51.24, 52.79, 55.37],
                [53.12, 60.78, 58.07],
                [70.8, 66.06, 56.33]
            ],
            [
                [50.03, 49.1, 48.9],
                [49.93, 49.88, 49.78],
                [51.33, 47.13, 41.48],
                [50.63, 51.03, 50.33],
                [49.16, 49.21, 51.04],
                [52.43, 54.41, 55.14],
                [56.08, 50.03, 64.18]
            ],
            [
                [51.4, 50.05, 53.35],
                [55.75, 60.92, 61.59],
                [65.76, 71.35, 60.99],
                [61.74, 67.38, 68.24],
                [72.02, 74.52, 83.25],
                [90.85, 90.8, 83.2]
            ],
            [
                [51.15, 48.65, 51.87],
                [50.90, 50.7, 52.06],
                [55.34, 51.17, 51.78],
                [51.91, 51.96, 50.60],
                [53.35, 59.55, 54.2],
            ],
            [
                [52.9, 52.49, 49.36],
                [57.36, 56.01, 54.29],
                [51.62, 58.08, 55.88],
                [67.32, 70.99, 60.05]
            ],
            [
                [51.02, 51.12, 50.92],
                [53.42, 50.50, 50.6],
                [52.41, 54.02, 52.41]
            ],
            [
                [49.54, 49.38, 49.38],
                [49.29, 49.24, 49.39]
            ],
            [
                [50.1, 49.85, 49.85]
            ]
        ]

        raw_data_loss = [
            [52.56, 56.11, 61.73, 59.91, 57.7, 83.25, 84.07, 81.94, 85.51, 98.4],
            [52.56, 55.37, 50.6, 55.97, 79.34, 88.53, 83.97, 81.04, 96.67],
            [51.89, 47.61, 59.95, 64.5, 79.35, 75.67, 85.25, 93.88],
            [50.51, 51.55, 53.56, 67.96, 66.36, 84.08, 90.47],
            [52.08, 62.8, 77.01, 70.3, 85.99, 95.85],
            [47.83, 50.13, 71.48, 62.82, 86.9],
            [57.09, 66.22, 73.07, 88.55],
            [55.68, 58.78, 85.23],
            [50.03, 73.89],
            [63.5]
        ]
    else:
        raise ValueError("Unknown filter: {}".format(args.filter))

    fill_data_new = np.zeros((len(targets), len(targets)))
    if args.mode == "meta":
        for i in range(len(targets)):
            for j in range(len(targets)-(i+1)):
                m, s = np.mean(raw_data_meta[i][j]), np.std(raw_data_meta[i][j])
                fill_data[i][j+i+1] = m
                # fill_data_new[len(targets) - (i + 1)][] = m
                mask[i][j+i+1] = False
                annot_data[i][j+i+1] = r'%d $\pm$ %d' % (m, s)

        # Rearrange data according to 1 - ratio
        fill_data = np.rot90(fill_data, 2).T
        annot_data = np.rot90(annot_data, 2).T

        sns_plot = sns.heatmap(fill_data, xticklabels=targets,
                               yticklabels=targets, annot=annot_data,
                               mask=mask, fmt="^")

    else:
        # TODO: Rearrange data according to 1 - ratio
        for i in range(len(targets)):
            for j in range(len(targets)-(i+1)):
                m, s = np.mean(raw_data_threshold[i][j]), np.std(
                    raw_data_threshold[i][j])
                fill_data[j+i+1][i] = m
                mask[j+i+1][i] = False
                annot_data[j+i+1][i] = r'%d $\pm$ %d' % (m, s)

        for i in range(len(targets)):
            for j in range(len(targets)-(i+1)):
                m = raw_data_loss[i][j]
                fill_data[i][j+i+1] = m
                mask[i][j+i+1] = False
                annot_data[i][j+i+1] = r'%d' % m

        for i in range(len(targets)):
            fill_data[i][i] = 0
            mask[i][i] = False
            annot_data[i][i] = "N.A."

        sns_plot = sns.heatmap(fill_data, xticklabels=targets,
                               yticklabels=targets, annot=annot_data,
                               mask=mask, fmt="^",
                               vmin=50, vmax=100)

        # mask = np.zeros_like(mask)
        # for i in range(len(targets)):
        #     for j in range(len(targets)-(i+1)):
        #         m = eff_vals[i][j]
        #         if m > 100:
        #             fill_data[i][j+i+1] = 100
        #             annot_data[i][i] = r'$\dagger$'
        #         else:
        #             fill_data[i][j+i+1] = m
        #             annot_data[i][i] = "%.1f" % m
        #         mask[i][j+i+1] = False
    # sns_plot.set(xlabel=r'$\alpha_0$', ylabel=r'$\alpha_1$')
    # sns_plot.figure.savefig("./plots/meta_heatmap_%s_%s.pdf" % (args.filter, args.mode))
    # exit(0)

    #track max values
    max_values = np.zeros_like(fill_data)
    eff_vals = np.zeros_like(fill_data)
    wanted = []
    for i in range(len(targets)):
        for j in range(len(targets)-(i+1)):

            n_eff_loss = get_n_effective(
                max(raw_data_meta[i][j]) / 100, float(targets[i]), float(targets[i+j+1]))
            if float(targets[i]) == 0.5:
                wanted.append(n_eff_loss)
                # print(n_eff_loss, targets[i+j+1])
            if float(targets[i+j+1]) == 0.5:
                wanted.append(n_eff_loss)
                # print(n_eff_loss, targets[i])

            max_values[i][j+i+1] = max(raw_data_loss[i]
                                       [j], max(raw_data_threshold[i][j]))
            max_values[i][j+i+1] = max(max_values[i]
                                       [j], max(raw_data_meta[i][j]))
            n_eff = get_n_effective(
                max_values[i][j+i+1] / 100, float(targets[i]), float(targets[i+j+1]))
            eff_vals[i][j] = np.abs(n_eff)
            # print(targets[i], targets[i+j+1], eff_vals[i][j],
            #       bound(float(targets[i]), float(targets[j]), n_eff))
    print("Median", np.median(wanted))

    # for i in range(len(targets)):
    #     print(['%.2f' % x for x in eff_vals[i][:len(targets)-(i+1)]])
    # for a in fill_data:
    #     a_wanted = [x for x in a if x > 0]
    #     if len(a_wanted) == 0:
    #         continue
    #     print("[" + ",".join(["r'%d $\pm$ %d'" % (np.mean(x), np.std(x))
    #                           for x in a_wanted]) + "]")

    # Rearrange data according to 1 - ratio
    # for i, ev in enumerate(eff_vals[:-1]):
    #     for j, x in enumerate(ev[:-(i+1)]):
    #         print(targets[(len(targets) - 1) - i], targets[(len(targets) - 1) - (j+i+1)], "%.2f" % x)
    #     print()

    eff_vals_mean = eff_vals.flatten()
    eff_vals_mean = eff_vals_mean[eff_vals_mean != np.inf]
    eff_vals_mean = eff_vals_mean[eff_vals_mean > 0]

    print("Mean N_leaked value: %.2f" % np.mean(eff_vals_mean))
    print("Median N_leaked value: %.2f" % np.median(eff_vals_mean))

    sns_plot.set(xlabel=r'$\alpha_0$', ylabel=r'$\alpha_1$')
    sns_plot.figure.savefig("./plots/meta_heatmap_%s_%s.pdf" % (args.filter, args.mode))
