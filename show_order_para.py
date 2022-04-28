import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def read_phi(fin, t0=0):
    with open(fin, "r") as f:
        lines = f.readlines()
        n = len(lines)
        phi, theta = np.zeros((2, n))
        for i, line in enumerate(lines):
            s = line.rstrip("\n").split("\t")
            phi[i] = float(s[0])
            theta[i] = float(s[1])
    t = (np.arange(n) + 1) * 100 + t0
    return t, phi, theta


def read_order_para(f0):
    str_list = f0.split("_")
    prefix = ""
    for s in str_list[:-1]:
        prefix += "%s_" % s
    files = glob.glob("%s*.dat" % prefix)
    t_beg_list = sorted([int(f.rstrip(".dat").split("_")[-1]) for f in files])

    phi_list, theta_list = [], []
    for t_beg in t_beg_list:
        fin = "%s%d.dat" % (prefix, t_beg)
        with open(fin, "r") as f:
            lines = f.readlines()
            n = len(lines)
            lines = lines[:n // 100 * 100]
            for line in lines:
                s = line.rstrip("\n").split("\t")
                phi_list.append(float(s[0]))
                theta_list.append(float(s[1]))
    phi_arr = np.array(phi_list)
    theta_arr = np.array(theta_list)
    t = (np.arange(phi_arr.size) + 1) * 100
    return t, phi_arr, theta_arr


def untangle(theta):
    theta_new = np.zeros_like(theta)
    theta_new[0] = theta[0]
    phase = 0.
    threshold = np.pi
    offset = 2 * np.pi
    for i in range(1, theta.size):
        d_theta = theta[i] - theta[i - 1]
        if d_theta > threshold:
            phase -= offset
        elif d_theta < -threshold:
            phase += offset
        theta_new[i] = theta[i] + phase
    return theta_new


def get_title(f0):
    s = os.path.basename(f0).rstrip(".dat").split("_")
    if len(s) == 8:
        Lx = int(s[0])
        Ly = int(s[1])
        D = float(s[2])
        rho_0 = float(s[3])
        v0 = float(s[4])
        seed = int(s[5])
        dt = float(s[6])
        # t_beg = int(s[7])
        title = f"$L_x={Lx},L_y={Ly},D={D:g}\\rho_0={rho_0:g},v_0={v0:g}," \
            f"dt={dt:g},{{\\rm seed}}={seed}$"
    elif len(s) == 7:
        L = int(s[0])
        D = float(s[1])
        rho_0 = float(s[2])
        v0 = float(s[3])
        seed = int(s[4])
        dt = float(s[5])
        title = f"$L={L},D={D:g}\\rho_0={rho_0:g},v_0={v0:g}," \
            f"{{\\rm seed}}={seed}$"
    return title


if __name__ == "__main__":
    os.chdir("series")
    f0 = "1200_160_0.300_1.200_1.0_4001_0.1_0.dat"

    t, phi, theta = read_order_para(f0)
    # theta = untangle(theta)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax1.plot(t, phi)
    ax2.plot(t, theta/np.pi, "-", ms=0.5)
    ax1.set_ylabel(r"$m$")
    ax2.set_ylabel(r"$\theta_m/\pi$")
    ax2.set_xlabel(r"$t$")
    ax1.set_xlim(2.55e8, 2.6e8)
    plt.suptitle(get_title(f0))
    plt.show()
    plt.close()
