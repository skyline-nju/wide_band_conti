import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from read_fields import read_last_frame


def get_D_rho0(f):
    basename = os.path.basename(f)
    s = basename.split("_")
    D = float(s[2])
    rho0 = float(s[3])
    return D, rho0


def get_D_rho0_range(files):
    D_arr = []
    rho0_arr = []
    for f in files:
        D, rho0 = get_D_rho0(f)
        if D not in D_arr:
            D_arr.append(D)
        if rho0 not in rho0_arr:
            rho0_arr.append(rho0)
    return np.array(sorted(D_arr)), np.array(sorted(rho0_arr))


def get_files(seed_in, Lx, Ly, dx, D=None, rho0=None, seed_mod="full"):
    def find_files(Lx, Ly, seed, dx, D, rho0):
        if D is None and rho0 is None:
            pat = f"fields/{Lx}_{Ly}_*_1.0_{seed}_{dx}_*_0.bin"
        elif D is not None and rho0 is not None:
            pat = f"fields/{Lx}_{Ly}_{D:.3f}_{rho0:.3f}" \
                f"_1.0_{seed}_{dx}_*_0.bin"
        elif D is not None:
            pat = f"fields/{Lx}_{Ly}_{D:.3f}_*_1.0_{seed}_{dx}_*_0.bin"
        else:
            pat = f"fields/{Lx}_{Ly}_*_{rho0:.3f}_1.0_{seed}_{dx}_*_0.bin"
        return glob.glob(pat)

    if not isinstance(seed_in, list):
        seeds = [seed_in]
    else:
        seeds = seed_in
    files = []
    if seed_mod == "full":
        for seed in seeds:
            files.extend(find_files(Lx, Ly, seed, dx, D, rho0))
    elif seed_mod == "max":
        seed_exist = []
        for seed_i in seeds:
            fs = find_files(Lx, Ly, seed_i, dx, D, rho0)
            if len(fs) > 0:
                seed_exist.append(seed_i)
        if len(seed_exist) > 0:
            seed_max = max(seed_exist)
            files = find_files(Lx, Ly, seed_max, dx, D, rho0)
    return files


def near_liquid_binodal(D, rho0):
    if (D == 0.25 and rho0 < 2.8) or \
        (D == 0.26 and rho0 < 3.0) or \
        (D == 0.27 and rho0 < 3.0) or \
        (D == 0.28 and rho0 < 3.0) or \
        (D == 0.29 and rho0 < 3.2) or \
        (D == 0.30 and rho0 < 3.4) or \
        (D == 0.31 and rho0 < 3.4) or \
        (D == 0.32 and rho0 < 3.4) or \
        (D == 0.33 and rho0 < 3.4) or \
        (D == 0.34 and rho0 < 3.4) or \
        (D == 0.35 and rho0 < 3.6) or \
        (D == 0.36 and rho0 < 3.8) or \
        (D == 0.37 and rho0 < 4.0) or \
        (D == 0.38 and rho0 < 4.2) or \
            (D >= 0.39):
        res = False
    else:
        res = True
    return res


def plot_full_PD():
    Lx = 1200
    Ly = 300
    dx = 2
    seed1 = 2001
    seed2 = 4001
    seeds = [seed1, seed2]
    files = get_files(seeds, Lx, Ly, dx)
    D_arr, rho0_arr = get_D_rho0_range(files)
    rho0_list = []
    for rho0 in rho0_arr:
        if np.mod(int(np.round(rho0 * 10)), 2) == 0 and np.mod(
                int(np.round(rho0 * 100)), 10) == 0:
            rho0_list.append(rho0)
    rho0_arr = np.array(rho0_list)
    figsize = (rho0_arr.size * 0.3 * 4, D_arr.size * 0.3 + 2)
    fig, axes = plt.subplots(D_arr.size,
                             rho0_arr.size,
                             sharex=True,
                             sharey=True,
                             figsize=figsize)

    for j, D in enumerate(D_arr):
        for i, rho0 in enumerate(rho0_arr):
            ax = axes[D_arr.size - j - 1][i]
            flag_red_frame = False
            fs_a = get_files(4001, Lx, Ly, dx, D, rho0)
            fs_b = get_files(2001, Lx, Ly, dx, D, rho0)
            if len(fs_a) > 0 and len(fs_b) > 0:
                if not near_liquid_binodal(D, rho0):
                    fs = fs_a
                else:
                    fs = fs_b
                    flag_red_frame = True
            elif len(fs_a) > 0:
                fs = fs_a
            elif len(fs_b) > 0:
                if near_liquid_binodal(D, rho0):
                    fs = fs_b
                    flag_red_frame = True
                else:
                    fs = []
            else:
                fs = []

            if len(fs) > 0:
                frame = read_last_frame(fs[0])
                t, rho, mx, my = frame
                ax.imshow(rho, origin="lower", vmax=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if flag_red_frame:
                    ax.spines['top'].set_color('tab:red')
                    ax.spines['bottom'].set_color('tab:red')
                    ax.spines['left'].set_color('tab:red')
                    ax.spines['right'].set_color('tab:red')
                    ax.spines["top"].set(linewidth=2)
                    ax.spines["bottom"].set(linewidth=2)
                    ax.spines["left"].set(linewidth=2)
                    ax.spines["right"].set(linewidth=2)
            else:
                ax.spines['top'].set_color('none')
                ax.spines['bottom'].set_color('none')
                ax.spines['left'].set_color('none')
                ax.spines['right'].set_color('none')
            if i == 0 and j % 2 == 0:
                ax.set_ylabel(r"$%.2f$" % D)
            if j == 0:
                ax.set_xlabel(r"$%.1f$" % rho0)
    plt.tight_layout(h_pad=0.001, w_pad=0.0001)
    # plt.show()
    plt.savefig("PD.png")
    plt.close()


def plot_const_D(D, seed=4001, Lx=1200, Ly=300, dx=2):
    files = get_files(seed, Lx, Ly, dx, D)
    D_arr, rho0_arr = get_D_rho0_range(files)
    figsize = (4 * 0.8, rho0_arr.size * 0.8)
    # mask = rho0_arr <= 0.6
    # rho0_arr = rho0_arr[mask]
    # figsize = (4 * 3.2, rho0_arr.size * 3.2)

    fig, axes = plt.subplots(rho0_arr.size,
                             1,
                             sharex=True,
                             sharey=True,
                             figsize=figsize,
                             constrained_layout=True)
    for i, rho0 in enumerate(rho0_arr):
        if rho0_arr.size == 1:
            ax = axes
        else:
            ax = axes[i]
        fs = get_files(seed, Lx, Ly, dx, D, rho0, seed_mod="max")
        frame = read_last_frame(fs[0])
        t, rho, mx, my = frame
        ax.imshow(rho, origin="lower", vmax=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(r"$%g$" % rho0, fontsize="xx-large")
        ax.text(1,
                1,
                r"$t=%g$" % (t * 0.1),
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                color="w",
                fontsize=23)
    # plt.suptitle(r"$L_x=%d,L_y=%d,D=%g$" % (Lx, Ly, D))
    # plt.show()
    plt.savefig("a.png")
    plt.close()


def plot_gas_binadal():
    Lx = 1200
    Ly = 300
    dx = 2
    seeds = [4001, 4002]
    files = get_files(seeds, Lx, Ly, dx)
    D_arr, rho0_arr = get_D_rho0_range(files)
    print("D =", D_arr)
    print("rho0 =", rho0_arr)

    figsize = (rho0_arr.size * 0.3 * 4, D_arr.size * 0.3 + 2)
    fig, axes = plt.subplots(D_arr.size,
                             rho0_arr.size,
                             sharex=True,
                             sharey=True,
                             figsize=figsize)
    for j, D in enumerate(D_arr):
        for i, rho0 in enumerate(rho0_arr):
            fs = get_files(seeds, Lx, Ly, dx, D, rho0, "max")
            ax = axes[D_arr.size - j - 1][i]
            if len(fs) > 0:
                frame = read_last_frame(fs[0])
                t, rho, mx, my = frame
                ax.imshow(rho, origin="lower", vmax=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.spines['top'].set_color('none')
                ax.spines['bottom'].set_color('none')
                ax.spines['left'].set_color('none')
                ax.spines['right'].set_color('none')
            if i == 0:
                ax.set_ylabel(r"$%.2f$" % D, rotation="horizontal")
            if j == 0:
                ax.set_xlabel(r"$%.1f$" % rho0)
    plt.tight_layout(h_pad=0.001, w_pad=0.0001)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # plot_full_PD()
    plot_const_D(0.35, seed=[4001, 4002], Lx=2400, dx=3)
    # plot_const_D(0.35, seed=[1001, 1002])

    # plot_const_D(0.32, seed=[4001])
    # plot_gas_binadal()
