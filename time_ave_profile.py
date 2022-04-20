import numpy as np
import matplotlib.pyplot as plt
# import sys
from space_time import read_y_averaged_fields


def find_root(y, y0, ascent=True):
    idx = []
    if ascent:
        for i in range(y.size):
            if y[i - 1] <= y0 < y[i]:
                # if y[i - 10] < y0 and y[i - 20] < y0 and y[i - 30] < y0 and \
                #     y[(i+10) % y.size] > y0 and y[(i+20) % y.size] > y0 and \
                #         y[(i+30) % y.size] > y0:
                if y[i-5] < y0 and y[(i+5) % y.size] > y0:
                    idx.append(i)
    else:
        for i in range(y.size - 1, -1, -1):
            if y[i] <= y0 < y[i - 1]:
                # if y[(i+10) % y.size] < y0 and y[(i+20) % y.size] < y0 and \
                #     y[(i+30) % y.size] < y0 and y[i-10] > y0 and \
                #         y[i-20] > y0 and y[i-30] > y0:
                if y[(i+5) % y.size] < y0 and y[i-5] > y0:
                    idx.append((i - 1 + y.size) % y.size)
    return idx


def cal_time_ave_profiles(Lx,
                          Ly,
                          D,
                          rho0,
                          seed,
                          dx,
                          rho_thresh=2,
                          v0=1.,
                          beg=0,
                          end=None):
    t_arr, x, rho, mx, my = read_y_averaged_fields(Lx=Lx,
                                                   Ly=Ly,
                                                   D=D,
                                                   rho0=rho0,
                                                   seed=seed)
    rho_m, mx_m, my_m = np.zeros((3, rho.shape[1]))
    count = 0
    ascent = np.mean(mx[0]) < 0
    for j, t in enumerate(t_arr[beg:end]):
        print("frame", j)
        i = j + beg
        idx_list = find_root(rho[i], rho_thresh, ascent)
        if len(idx_list) > 1 or len(idx_list) == 0:
            plt.plot(x, rho[i])
            plt.axhline(rho_thresh, linestyle="dashed", c="tab:red")
            for idx in idx_list:
                plt.axvline(x[idx], linestyle="dashed", c="tab:red")
            plt.title(r"frame $%g, t=%g$" % (i, t))
            plt.show()
            plt.close()
        else:
            shift = Lx // 4 - idx_list[0]
            rho_m += np.roll(rho[i], shift)
            mx_m += np.roll(mx[i], shift)
            my_m += np.roll(my[i], shift)
            count += 1
    rho_m /= count
    mx_m /= count
    my_m /= count
    fnpz = f"space_time/time_ave/{Lx}_{Ly}_{D:.3f}_{rho0:.3f}_{v0:.1f}_" \
        f"{seed}.npz"
    np.savez_compressed(fnpz, x=x, rho_m=rho_m, mx_m=mx_m, my_m=my_m)


def get_time_ave_profile(Lx, Ly, D, rho0, seed, v0=1.):
    fnpz = f"space_time/time_ave/{Lx}_{Ly}_{D:.3f}_{rho0:.3f}_{v0:.1f}_" \
        f"{seed}.npz"
    with np.load(fnpz) as data:
        x = data["x"]
        rho_m = data["rho_m"]
        mx_m = data["mx_m"]
        my_m = data["my_m"]
    return x, rho_m, mx_m, my_m


def varied_Ly(Lx=1200, Lys=[160, 320, 640], D=0.3, rho0=1.4, seed=4001):
    fig, ax = plt.subplots(constrained_layout=True)
    for Ly in Lys:
        x, rho_m, mx_m, my_m = get_time_ave_profile(Lx, Ly, D, rho0, seed)
        if Ly == 160:
            x = Lx - x + 2
        ax.plot(x, rho_m, label="$L_y=%d$" % Ly)
    ax.set_xlim(0, Lx)
    ax.set_xlabel(r"$x$", fontsize="xx-large")
    ax.set_ylabel(r"$\langle\rho \rangle_{y,t}$", fontsize="xx-large")
    plt.legend(fontsize="x-large", loc="upper left")
    plt.suptitle(r"$L_x=%d,D=%g,\rho_0=%g$" % (Lx, D, rho0),
                 fontsize="xx-large")
    # plt.show()
    fout = "space_time/time_ave/varied_Ly_Lx%d_D%g_r%g.pdf" % (Lx, D, rho0)
    plt.savefig(fout)
    plt.close()


def varied_rho0(Lx=1200, Ly=640, D=0.3, seed=4001):
    if Lx == 1200 and Ly == 640 and D == 0.3 and seed == 4001:
        rho0s = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    fig, ax = plt.subplots(
        3, 1, constrained_layout=True, figsize=(6, 8), sharex=True)
    ax_in = ax[2].inset_axes([0.1, 0.22, 0.35, 0.75])
    for rho0 in rho0s:
        x, rho_m, mx_m, my_m = get_time_ave_profile(Lx, Ly, D, rho0, seed)
        ax[0].plot(x, rho_m, label=r"$\rho_0=%g$" % rho0)
        ax[1].plot(x, -mx_m)
        ax[2].plot(x, -mx_m/rho_m)
        ax_in.plot(x, -mx_m/rho_m)
        # ax[3].plot(x, my_m)
        # ax[4].plot(x, my_m/rho_m)

    ax[0].set_ylabel(r"$\langle\rho \rangle_{y,t}$", fontsize="x-large")
    ax[1].set_ylabel(r"$-\langle{\bf m}_x \rangle_{y,t}$", fontsize="x-large")
    ax[2].set_ylabel(r"$-\overline{{\bf p}}_x$", fontsize="x-large")
    # ax[3].set_ylabel(r"$\langle{\bf m}_y \rangle_{y,t}$", fontsize="x-large")
    # ax[4].set_ylabel(r"$\overline{{\bf p}}_y$", fontsize="x-large")
    ax[-1].set_xlabel(r"$x$", fontsize="x-large")
    ax[-1].set_xlim(0, Lx)
    ax_in.set_xlim(550, 1100)
    ax_in.set_ylim(0.65, 0.71)

    plt.suptitle(r"$L_x=%d,L_y=%d, D=%g$" % (Lx, Ly, D),
                 fontsize="x-large")
    ax[0].legend(loc="upper left", fontsize="large")
    # plt.show()
    fout = "space_time/time_ave/varied_rho0_L%d_%d_D%g.pdf" % (Lx, Ly, D)
    plt.savefig(fout)
    plt.close()


def inverse_bands():
    Lx = 1200
    Ly = 640
    D = 0.3

    rho0_list = [1.8, 3.4, 3.5]
    fig, ax = plt.subplots(
        3, 1, constrained_layout=True, figsize=(6, 8), sharex=True)
    for rho0 in rho0_list:
        if rho0 == 1.8:
            seed = 4001
            marker = "--"
        else:
            seed = 2001
            marker = "-"
        x, rho_m, mx_m, my_m = get_time_ave_profile(Lx, Ly, D, rho0, seed)

        if seed == 2001:
            x = Lx - x + 2
        else:
            mx_m = -mx_m
            my_m = -my_m
        ax[0].plot(x, rho_m, marker, label=r"$\rho_0=%g$" % rho0)
        ax[1].plot(x, mx_m, marker)
        ax[2].plot(x, mx_m/rho_m, marker)
    ax[0].set_ylabel(r"$\langle\rho \rangle_{y,t}$", fontsize="x-large")
    ax[1].set_ylabel(r"$|\langle{\bf m}_x \rangle_{y,t}|$", fontsize="x-large")
    ax[2].set_ylabel(r"$|\overline{{\bf p}}_x|$", fontsize="x-large")

    plt.suptitle(r"$L_x=%d,L_y=%d, D=%g$" % (Lx, Ly, D),
                 fontsize="x-large")
    ax[0].legend(loc="lower left", fontsize="large")
    # plt.show()
    fout = "space_time/time_ave/inv_band_L%d_%d_D%g.pdf" % (Lx, Ly, D)
    plt.savefig(fout)
    plt.close()


if __name__ == "__main__":
    Lx = 1200
    Ly = 640
    D = 0.3
    rho0 = 3.4
    seed = 2001
    rho_thresh = 2
    dx = 2

    # beg = 400
    # end = None
    # cal_time_ave_profiles(Lx, Ly, D, rho0, seed, dx, beg=beg, end=end)

    # x, rho_m, mx_m, my_m = get_time_ave_profile(Lx, Ly, D, rho0, seed)

    # plt.plot(x, rho_m)
    # plt.show()
    # plt.close()

    # varied_Ly()
    # varied_rho0()
    inverse_bands()
