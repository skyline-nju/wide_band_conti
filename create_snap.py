import numpy as np
import os
from plot_snap import read_snap, plot_snap, get_para


def save_to_file(x, y, theta, para, adjust_rho0=False):
    n_new = int(para["rho0"] * para["Lx"] * para["Ly"])

    if n_new < x.size:
        n_old = x.size
        tmp = np.array([x, y, theta]).T
        np.random.shuffle(tmp)
        tmp = tmp[:n_new, :]
        x, y, theta = tmp.T
        print("Warning, remove", n_old-n_new, "particles")
    elif n_new > x.size:
        n2 = n_new - x.size
        x2 = np.random.rand(n2) * para["Lx"]
        y2 = np.random.rand(n2) * para["Ly"]
        theta2 = np.random.rand(n2) * np.pi * 2
        x = np.hstack((x, x2))
        y = np.hstack((y, y2))
        theta = np.hstack((theta, theta2))
        print("Warning, add", n2, "particles")

    if adjust_rho0:
        para["rho0"] = x.size / (para["Lx"] * para["Ly"])
    elif x.size != int(para["rho0"] * para["Lx"] * para["Ly"]):
        print("Waring, particle number =", x.size)

    if not os.path.exists("snap/built"):
        os.mkdir("snap/built")
    if para["Lx"] == para["Ly"]:
        fout = "snap/built/s%d_%.3f_%.3f_%.1f_%d_%.1f_%08d.bin" % (
            para["Lx"], para["D"], para["rho0"], para["v0"],
            para["seed"], para["dt"], para["t"])
    else:
        fout = "snap/built/s%d_%d_%.3f_%.3f_%.1f_%d_%.1f_%08d.bin" % (
            para["Lx"], para["Ly"], para["D"], para["rho0"],
            para["v0"], para["seed"], para["dt"], para["t"])
    print("output new snapshot to", fout)
    data = np.zeros((3, n_new), np.float32)
    data[0] = x
    data[1] = y
    data[2] = theta
    data = data.T.flatten()
    data.tofile(fout)


def duplicate(fin, nx=1, ny=1, rot_90_CW=False):
    para0 = get_para(fin)
    x0, y0, theta0 = read_snap(fin)
    if rot_90_CW:
        x0, y0 = y0, -x0
        y0 += para0["Lx"]
        theta0 += np.pi / 2
    para_new = {key: para0[key] for key in para0}
    para_new["Lx"] *= nx
    para_new["Ly"] *= ny
    para_new["seed"] = int("2%d" % para0["seed"])
    para_new["t"] = 0
    N0 = x0.size
    N_new = int(para0["rho0"] * para_new["Lx"] * para_new["Ly"])
    print("N_old=%d, N_new=%d, N_new%%N_old=%d" % (N0, N_new, N_new % N0))

    if N_new < N0 * nx * ny:
        size_new = N0 * nx * ny
    else:
        size_new = N_new
    x, y, theta = np.zeros((3, size_new), np.float32)
    for row in range(ny):
        dy = row * para0["Ly"]
        for col in range(nx):
            dx = col * para0["Lx"]
            k = col + row * nx
            beg = k * N0
            end = beg + N0
            x[beg:end] = x0 + dx
            y[beg:end] = y0 + dy
            theta[beg:end] = theta0
    if N_new < size_new:
        x = x[:N_new]
        y = y[:N_new]
        theta = theta[:N_new]
    elif N_new > N0 * nx * ny:
        for j in range(N0 * nx * ny, N_new):
            x[j] = np.random.rand() * para_new["Lx"]
            y[j] = np.random.rand() * para_new["Ly"]
            theta[j] = np.random.rand() * np.pi * 2.
    plot_snap(x0, y0, theta0, para0, frac=0.01)
    plot_snap(x, y, theta, para_new, frac=0.01 / (nx * ny))

    if not os.path.exists("snap/built"):
        os.mkdir("snap/built")
    if para_new["Lx"] == para_new["Ly"]:
        fout = "snap/built/s%d_%.3f_%.3f_%.1f_%d_%.1f_%08d.bin" % (
            para_new["Lx"], para_new["D"], para_new["rho0"], para_new["v0"],
            para_new["seed"], para_new["dt"], para_new["t"])
    else:
        fout = "snap/built/s%d_%d_%.3f_%.3f_%.1f_%d_%.1f_%08d.bin" % (
            para_new["Lx"], para_new["Ly"], para_new["D"], para_new["rho0"],
            para_new["v0"], para_new["seed"], para_new["dt"], para_new["t"])

    print("output new snapshot to", fout)
    data = np.zeros((3, N_new), np.float32)
    data[0] = x
    data[1] = y
    data[2] = theta
    data = data.T.flatten()
    data.tofile(fout)


def slice(x, y, theta, rect, shift=False):
    xmin, xmax, ymin, ymax = rect
    mask = y >= ymin
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = y < ymax
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = x >= xmin
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    mask = x < xmax
    x = x[mask]
    y = y[mask]
    theta = theta[mask]
    if shift:
        x -= xmin
        y -= ymin
    return x, y, theta


def inverse(x, y, theta, xc, yc):
    x_inv = 2 * xc - x
    y_inv = 2 * yc - y
    theta_inv = theta + np.pi
    return x_inv, y_inv, theta_inv


def make_band_lane(x, y, theta, n_lane, width, length, direction="y"):
    if direction == "y":
        x_inv, y_inv, theta_inv = inverse(x, y, theta, length / 2, width / 2)
    else:
        x_inv, y_inv, theta_inv = inverse(x, y, theta, width / 2, length / 2)

    n_new = x.size * n_lane
    x_new, y_new, theta_new = np.zeros((3, n_new), np.float32)
    for i_lane in range(n_lane):
        i = i_lane * x.size
        j = (i_lane + 1) * x.size
        if i_lane % 2 == 0:
            x_new[i:j] = x
            y_new[i:j] = y
            theta_new[i:j] = theta
        else:
            x_new[i:j] = x_inv
            y_new[i:j] = y_inv
            theta_new[i:j] = theta_inv
        if direction == "y":
            y_new[i:j] += i_lane * width
        else:
            x_new[i:j] += i_lane * width
    return x_new, y_new, theta_new


def create_band_lane_snap():
    # fin = "snap/s2400_0.290_1.000_0.5_133_47200000.bin"
    fin = "snap/s2400_0.350_1.000_0.5_411_26160000.bin"
    para = get_para(fin)
    if para["seed"] == 133:
        direction = "y"
        length = para["Lx"]
        # width, ymin, ymax = 400, 1500, 1900
        width, ymin, ymax = 600, 0, 600
        rect = [0, para["Lx"], ymin, ymax]
    else:
        direction = "x"
        length = para["Ly"]
        width, xmin, xmax = 400, 1500, 1900
        rect = [xmin, xmax, 0, para["Ly"]]
    x, y, theta = read_snap(fin)
    x, y, theta = slice(x, y, theta, rect, shift=True)
    n_lane = 6
    para["Ly"] = n_lane * width
    x, y, theta = make_band_lane(x, y, theta, n_lane, width, length, direction)

    rho_new = float("%.3f" % (x.size / (para["Lx"] * para["Ly"])))
    n_new = int(rho_new * para["Lx"] * para["Ly"])
    print("n=", x.size, "rho=", x.size / (para["Lx"] * para["Ly"]))

    if n_new <= x.size:
        x = x[:n_new]
        y = y[:n_new]
        theta = theta[:n_new]
    else:
        n2 = n_new - x.size
        x2 = np.random.rand(n2) * para["Lx"]
        y2 = np.random.rand(n2) * para["Ly"]
        theta2 = np.random.rand(n2) * np.pi * 2
        x = np.hstack((x, x2))
        y = np.hstack((y, y2))
        theta = np.hstack((theta, theta2))
        print("Warning, add", n2, "particles")

    print("n=", x.size, "rho=", x.size / (para["Lx"] * para["Ly"]))
    para["rho0"] = rho_new
    para["seed"] = int("%d%d%d" % (2, n_lane, para["seed"]))
    para["t"] = 0
    print("particle number calculated from rho0",
          int(para["rho0"] * para["Lx"] * para["Ly"]))
    plot_snap(x, y, theta, para, frac=0.01, show_relative_angle=False)

    if not os.path.exists("snap/slice"):
        os.mkdir("snap/slice")

    fout_subfix = "{eta:.3f}_{rho0:.3f}_{v0:.1f}_{seed}_{t:08d}.bin".format(
        **para)
    if para["Lx"] == para["Ly"]:
        fout = f"snap/slice/s{para['Lx']}_{fout_subfix}"
    else:
        fout = f"snap/slice/s{para['Lx']}_{para['Ly']}_{fout_subfix}"
    print("output new snapshot to", fout)
    data = np.zeros((3, n_new), np.float32)
    data[0] = x
    data[1] = y
    data[2] = theta
    data = data.T.flatten()
    data.tofile(fout)


if __name__ == "__main__":
    fin = r"snap/built/s1200_5120_0.300_1.600_1.0_184001_0.1_00000000.bin"
    # fin = r"snap/s1200_1280_0.300_2.000_1.0_1001_0.1_05410000.bin"

    x, y, theta = read_snap(fin)
    para = get_para(fin)
    print(x.size, para["Lx"] * para["Ly"] * para["rho0"])
    plot_snap(x, y, theta, para, frac=0.005)

    # mask = y < 20
    # x = np.hstack((x, x[mask]))
    # y = np.hstack((y, y[mask]+300))
    # theta = np.hstack((theta, theta[mask]))
    # para["Ly"] = 320
    # para["t"] = 0
    # plot_snap(x, y, theta, para, frac=0.05)

    # x -= 180
    # x[x < 0] += 1200
    # plot_snap(x, y, theta, para, frac=0.005)

    # mask = x >= 1200
    # xg = x[mask]
    # yg = y[mask]
    # theta_g = theta[mask]

    # x = np.hstack((x, xg + 1200))
    # y = np.hstack((y, yg))
    # theta = np.hstack((theta, theta_g))
    # para["Lx"] = 3600
    # # para["rho0"] = x.size / (para["Lx"] * para["Ly"])
    # para["rho0"] = 0.923
    # para["seed"] = 384001
    # para["t"] = 0
    # plot_snap(x, y, theta, para, frac=0.005)

    x = np.hstack((x, x))
    y = np.hstack((y, y+5120))
    theta = np.hstack((theta, theta))

    para["t"] = 0
    para["seed"] = 1164001
    para["Ly"] = 5120 * 2
    # # para["rho0"] = 0.8
    plot_snap(x, y, theta, para, frac=0.005)

    # para["t"] = 0
    # para["rho0"] = 2.2
    # # para["seed"] = 1001
    # plot_snap(x, y, theta, para, frac=0.005)

    save_to_file(x, y, theta, para)
