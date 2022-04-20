import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    D = np.array([
        0.16, 0.18, 0.20, 0.22, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.33,
        0.34, 0.35
    ])
    width = np.array(
        [10, 12, 15, 34, 65, 75, 125, 140, 140, 190, 250, 270, 310, 370]) * 2
    plt.plot(D, width, "o")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(r"$D$", fontsize="x-large")
    plt.ylabel("max width for a single band", fontsize="x-large")
    plt.title(r"$L_x=1200, L_y=300$", fontsize="x-large")
    plt.tight_layout()
    # plt.show()
    plt.savefig("band_width.pdf")
    plt.close()
