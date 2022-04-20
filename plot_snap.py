import numpy as np
import matplotlib.pyplot as plt
import struct
import os


def read_snap(fin):
    with open(fin, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        f.seek(0)
        N = filesize // 12
        buf = f.read()
        data = struct.unpack("%df" % (N * 3), buf)
        x, y, theta = np.array(data, np.float32).reshape(N, 3).T
    return x, y, theta


def zoom_in(x0, y0, theta0, xmin, xmax, ymin, ymax):
    if xmin < xmax:
        mask = x0 > xmin
        x = x0[mask]
        y = y0[mask]
        theta = theta0[mask]
        mask = x < xmax
        x = x[mask]
        y = y[mask]
        theta = theta[mask]
    else:
        mask = x0 > xmin
        x1 = x0[mask]
        y1 = y0[mask]
        theta1 = theta0[mask]
        mask = x0 < xmax
        x2 = x0[mask]
        y2 = y0[mask]
        theta2 = theta0[mask]
        x = np.hstack([x1, x2])
        y = np.hstack([y1, y2])
        theta = np.hstack([theta1, theta2])

    if ymin < ymax:
        mask = y > ymin
        x = x[mask]
        y = y[mask]
        theta = theta[mask]
        mask = y < ymax
        x = x[mask]
        y = y[mask]
        theta = theta[mask]
    else:
        mask = y > ymin
        x1 = x[mask]
        y1 = y[mask]
        theta1 = theta[mask]
        mask = y < ymax
        x2 = x[mask]
        y2 = y[mask]
        theta2 = theta[mask]
        x = np.hstack([x1, x2])
        y = np.hstack([y1, y2])
        theta = np.hstack([theta1, theta2])
    return x, y, theta


def random_select(x, y, theta, frac, idx_arr, rng=None):
    n = int(frac * x.size)
    if rng is not None:
        rng.shuffle(idx_arr)
    mask = idx_arr[:n]
    return x[mask], y[mask], theta[mask]


def get_para(fname):
    basename = os.path.basename(fname)
    s = basename.lstrip("s").rstrip(".bin").split("_")
    para = {}

    if len(s) == 7:
        para["Lx"] = int(s[0])
        para["Ly"] = para["Lx"]
        para["D"] = float(s[1])
        para["rho0"] = float(s[2])
        para["v0"] = float(s[3])
        para["seed"] = int(s[4])
        para["dt"] = float(s[5])
        para["t"] = int(s[6])
    elif len(s) == 8:
        para["Lx"] = int(s[0])
        para["Ly"] = int(s[1])
        para["D"] = float(s[2])
        para["rho0"] = float(s[3])
        para["v0"] = float(s[4])
        para["seed"] = int(s[5])
        para["dt"] = float(s[6])
        para["t"] = int(s[7])
    return para


def get_title(para):
    title_subfix = f"D={para['D']:g},\\rho_0={para['rho0']:g}," \
        f"v_0={para['v0']:g},{{\\rm seed}}={para['seed']},t={para['t']}$"
    if para["Lx"] == para["Ly"]:
        title = f"$L={para['Lx']},{title_subfix}"
    else:
        title = f"$L_x={para['Lx']}, L_y={para['Ly']},{title_subfix}"
    return title


def plot_snap(x=None,
              y=None,
              theta=None,
              para=None,
              fin=None,
              frac=1.,
              idx_arr=None,
              show_relative_angle=True):
    if fin is not None:
        para = get_para(fin)
        x, y, theta = read_snap(fin)
        n = int(para['rho0'] * para["Lx"] * para["Ly"])
        if n != x.size:
            print("Waring,", fin, "has", x.size, "particles, but should be", n,
                  "since rho0=", para["rho"])
    if para is not None:
        xmin, xmax = 0, para["Lx"]
        ymin, ymax = 0, para["Ly"]
        Lx, Ly = para["Lx"], para["Ly"]
    else:
        import math
        xmin, xmax = x.min(), x.max()
        ymin, ymax = math.floor(y.min()), math.ceil(y.max())
        Lx, Ly = xmax - xmin, ymax - ymin
        print(f"Lx={Lx}, Ly={Ly}")
    vx_m = np.mean(np.cos(theta))
    vy_m = np.mean(np.sin(theta))
    theta_m = np.arctan2(vy_m, vx_m)
    if frac < 1.:
        if idx_arr is None:
            rng = np.random.default_rng()
            idx_arr = np.arange(x.size)
            rng.shuffle(idx_arr)
        x1, y1, theta1 = random_select(x, y, theta, frac, idx_arr)
    else:
        x1, y1, theta1 = x, y, theta

    if show_relative_angle:
        c = theta1 - theta_m
        cb_label = r"$\theta - \theta_m$"
    else:
        c = theta1
        cb_label = r"$\theta$"
    c[c > np.pi] -= np.pi * 2
    c[c < -np.pi] += np.pi * 2

    if Lx >= 4 * Ly:
        figsize = (14, 4)
        cb_frac = 0.03
    elif Lx == 2 * Ly:
        figsize = (8, 4)
        cb_frac = 0.05
    else:
        figsize = (10, 8.5)
        cb_frac = 0.1
    plt.figure(figsize=figsize)
    # plt.subplot(111, fc="k")
    sca = plt.scatter(x1, y1, s=0.25, c=c, cmap="hsv")
    plt.axis("scaled")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cb = plt.colorbar(sca, fraction=cb_frac, shrink=1)
    cb.set_label(cb_label, fontsize="x-large")
    if para is not None:
        title = get_title(para)
        plt.suptitle(title, fontsize="xx-large")
    plt.tight_layout()
    plt.show()
    # plt.savefig("D:/data/tmp2/%04d.png" % i, dpi=300)
    plt.close()


if __name__ == "__main__":
    fin = r"snap/s1200_1280_0.300_2.000_1.0_1001_0.1_05060000.bin"
    x, y, theta = read_snap(fin)
    para = get_para(fin)
    print(x.size, para["Lx"] * para["Ly"] * para["rho0"])
    plot_snap(x, y, theta, para, frac=0.01)
