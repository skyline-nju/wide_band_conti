# import os
import numpy as np
import matplotlib.pyplot as plt
# import glob
# import sys
from read_fields import get_para, read_fields, get_nframe, read_last_frame


def vertical_profile():
    fin = "fields/2400_4800_0.350_1.500_1.0_243001_4_10000_0.1_0.bin"
    para = get_para(fin)
    nframes = get_nframe(fin, True)
    if isinstance(nframes, np.ndarray):
        nframes = np.sum(nframes)
    rho_t_y = np.zeros((nframes, para["Ly"] // para["dx"]))
    # y = np.arange(para["Ly"]//para["dx"]) * para["dx"] + para["dx"] / 2
    frames = read_fields(fin)

    for i, (t, rho, vx, vy) in enumerate(frames):
        rho_t_y[i] = np.mean(rho, axis=1)
    plt.imshow(rho_t_y.T, origin="lower", vmax=1.8)
    plt.colorbar()
    plt.show()
    plt.close()


def compare_profiles():
    f1 = "fields/1200_300_0.350_1.200_1.0_4001_2_10000_0.1_0.bin"
    f2 = "fields/1200_300_0.350_4.000_1.0_2001_2_10000_0.1_0.bin"

    para1 = get_para(f1)
    frame1 = read_last_frame(f1)
    frame2 = read_last_frame(f2)

    t1, rho1, vx1, vy1 = frame1
    t2, rho2, vx2, vy2 = frame2

    x = np.arange(rho1.shape[1]) * para1["dx"] + para1["dx"] / 2
    rho1_x = np.mean(rho1, axis=0)
    rho2_x = np.mean(rho2, axis=0)

    plt.plot(x, rho1_x)
    plt.plot(x, rho2_x)

    plt.show()
    plt.close()


if __name__ == "__main__":
    # fin = "fields/4800_300_0.350_4.100_1.0_1004_2_10000_0.1_0.bin"
    fin = "fields/4800_300_0.350_4.000_1.0_3001_3_10000_0.1_0.bin"
    # fin = "fields/4800_300_0.350_4.050_1.0_1004_2_10000_0.1_0.bin"
    # fin = "fields/4800_300_0.350_4.150_1.0_1003_3_10000_0.1_0.bin"
    para = get_para(fin)
    nframes = get_nframe(fin, True)
    if isinstance(nframes, np.ndarray):
        nframes = np.sum(nframes)
    rho_x = np.zeros((5, para["Lx"] // para["dx"]))
    frames = read_fields(fin, end=5)
    for i, (t, rho, vx, vy) in enumerate(frames):
        rho_x[i] = np.mean(rho, axis=0)
    x = (np.arange(rho_x[0].size) + 0.5) * para["dx"]

    fig, axes = plt.subplots(5,
                             1,
                             sharex=True,
                             sharey=True,
                             constrained_layout=True)

    for i, ax in enumerate(axes):
        ax.plot(x, rho_x[i])
        ax.axhline(3)
    plt.show()
    plt.close()

    # c0 = 768  # rho0=4.1, D=0.35
    c0 = 769
    fig, axes = plt.subplots(5,
                             1,
                             sharex=True,
                             sharey=True,
                             constrained_layout=True)
    for i, ax in enumerate(axes):
        shift = int(np.round(i * c0 / para["dx"]))
        rx = np.roll(rho_x[i], -shift)
        ax.plot(x, rx)
        ax.axhline(3)
    plt.show()
    plt.close()

    n_frames = nframes
    frames = read_fields(fin, end=n_frames)
    rho_x = np.zeros((n_frames, para["Lx"] // para["dx"]))
    for i, (t, rho, vx, vy) in enumerate(frames):
        rx = np.mean(rho, axis=0)
        shift = int(np.round(i * c0 / para["dx"]))
        rho_x[i] = np.roll(rx, -shift)
    plt.imshow(rho_x, origin="lower", vmax=4.5)
    # plt.colorbar()
    plt.show()
    plt.close()
