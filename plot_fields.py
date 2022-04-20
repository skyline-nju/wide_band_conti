import os
import numpy as np
import matplotlib.pyplot as plt
import glob
# import sys
from matplotlib.colors import hsv_to_rgb
from read_fields import get_para, read_fields


def map_v_to_rgb(theta, module, m_max=None):
    """
    Transform orientation and magnitude of velocity into rgb.

    Parameters:
    --------
    theta: array_like
        Orietation of velocity field.
    module: array_like
        Magnitude of velocity field.
    m_max: float, optional
        Max magnitude to show.

    Returns:
    --------
    RGB: array_like
        RGB corresponding to velocity fields.
    """
    H = theta / 360
    V = module
    if m_max is not None:
        V[V > m_max] = m_max
    V /= m_max
    S = np.ones_like(H)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def add_colorbar(ax, mmin, mmax, theta_min=0, theta_max=360, orientation="h"):
    """ Add colorbar for the RGB image plotted by plt.imshow() """
    V, H = np.mgrid[0:1:50j, 0:1:180j]
    if orientation == "v":
        V = V.T
        H = H.T
        box = [mmin, mmax, theta_min, theta_max]
    else:
        box = [theta_min, theta_max, mmin, mmax]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    ax.imshow(RGB, origin='lower', extent=box, aspect='auto')
    theta_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    if orientation == "h":
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'module $|{\bf m}|$', fontsize="large")
        ax.set_xlabel("orientation", fontsize="large")
    else:
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'orientation $\theta$', fontsize="large")
        # ax.set_xlabel(r"module $|{\bf m}|$", fontsize="large")
        ax.set_title(r"$|{\bf m}|$", fontsize="large")


def get_colobar_extend(vmin, vmax):
    if vmin is None or vmin == 0.:
        if vmax is None:
            ext = "neither"
        else:
            ext = "max"
    else:
        if vmax is None:
            ext = "min"
        else:
            ext = "both"
    return ext


def plot_density_momentum(rho, vx, vy, t, para, figsize=(12, 7.5), fout=None):
    theta = np.arctan2(vy, vx)
    theta[theta < 0] += np.pi * 2
    theta *= 180 / np.pi
    module = np.sqrt(vx**2 + vy**2)

    if para["Lx"] <= para["Ly"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    box = [0, para["Lx"], 0, para["Ly"]]
    if para["rho0"] <= 1.2:
        vmin1, vmax1 = None, 4
        vmin2, vmax2 = 0, 4
    elif para["rho0"] <= 4.5:
        vmin1, vmax1 = None, 6
        vmin2, vmax2 = 0, 6
    else:
        vmin1, vmax1 = 2, 12
        vmin2, vmax2 = 0, 8
    im1 = ax1.imshow(rho, origin="lower", extent=box, vmin=vmin1, vmax=vmax1)
    RGB = map_v_to_rgb(theta, module, m_max=vmax2)
    ax2.imshow(RGB, extent=box, origin="lower")

    vx_m, vy_m = np.mean(vx), np.mean(vy)
    phi = np.sqrt(vx_m**2 + vy_m**2) / para["rho0"]
    theta_m = np.arctan2(vy_m, vx_m) / np.pi * 180
    title_suffix = f"D={para['D']:g},\\rho_0={para['rho0']:g}," \
        f"v_0={para['v0']:g},{{\\rm seed}}={para['seed']},{{\\rm d}}t=" \
        f"{para['h']:g},t={t*para['h']:g},\\phi={phi:.3f}," \
        f"\\theta_m={theta_m:.1f}\\degree"
    if para["Lx"] == para["Ly"]:
        ax1.set_title(r"(a) density")
        ax2.set_title(r"(b) momentum")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        bbox1 = ax1.get_position().get_points().flatten()
        bbox2 = ax2.get_position().get_points().flatten()
        fig.subplots_adjust(bottom=0.24)
        bbox1[1], bbox1[3] = 0.14, 0.04
        bbox1[2] = bbox1[2] - bbox1[0] - 0.03
        bbox2[1], bbox2[3] = 0.08, 0.14
        bbox2[2] = bbox2[2] - bbox2[0]
        cb_ax1 = fig.add_axes(bbox1)
        cb_ax2 = fig.add_axes(bbox2)
        ext1 = get_colobar_extend(vmin1, vmax1)
        cb1 = fig.colorbar(im1,
                           ax=ax1,
                           cax=cb_ax1,
                           orientation="horizontal",
                           extend=ext1)
        cb1.set_label(r"density $\rho$", fontsize="x-large")
        add_colorbar(cb_ax2, vmin2, vmax2, 0, 360)
        title = r"$L=%d,%s$" % (para["Lx"], title_suffix)
        plt.suptitle(title, y=0.995, fontsize="x-large")
    elif para["Lx"] // para["Ly"] == 4:
        plt.tight_layout(rect=[0, -0.02, 1, 0.98])
        bbox1 = ax1.get_position().get_points().flatten()
        bbox2 = ax2.get_position().get_points().flatten()
        fig.subplots_adjust(right=0.92)
        bbox1[0], bbox1[2] = 0.91, 0.03
        bbox1[3] = bbox1[3] - bbox1[1]
        bbox2[0], bbox2[2] = 0.90, 0.045
        bbox2[3] = bbox2[3] - bbox2[1]
        cb_ax1 = fig.add_axes(bbox1)
        cb_ax2 = fig.add_axes(bbox2)

        ext1 = get_colobar_extend(vmin1, vmax1)
        cb1 = fig.colorbar(im1,
                           ax=ax1,
                           cax=cb_ax1,
                           orientation="vertical",
                           extend=ext1)
        cb1.set_label(r"density $\rho$", fontsize="x-large")
        add_colorbar(cb_ax2, vmin2, vmax2, 0, 360, "v")
    elif para["Lx"] // para["Ly"] == 2:
        plt.tight_layout(rect=[0, -0.01, 1, 0.98])
        bbox1 = ax1.get_position().get_points().flatten()
        bbox2 = ax2.get_position().get_points().flatten()
        fig.subplots_adjust(right=0.92)
        bbox1[0], bbox1[2] = 0.90, 0.04
        bbox1[3] = bbox1[3] - bbox1[1]
        bbox2[0], bbox2[2] = 0.89, 0.055
        bbox2[3] = bbox2[3] - bbox2[1]
        cb_ax1 = fig.add_axes(bbox1)
        cb_ax2 = fig.add_axes(bbox2)

        ext1 = get_colobar_extend(vmin1, vmax1)
        cb1 = fig.colorbar(im1,
                           ax=ax1,
                           cax=cb_ax1,
                           orientation="vertical",
                           extend=ext1)
        cb1.set_label(r"density $\rho$", fontsize="x-large")
        add_colorbar(cb_ax2, vmin2, vmax2, 0, 360, "v")
    else:
        # ax1.yaxis.set_tick_params(rotation=90)
        plt.tight_layout(rect=[-0.03, -0.01, 1.03, 0.99])
        title = r"$t=%g$" % (t * para["h"])
        plt.suptitle(title, y=0.999, fontsize="x-large")
    if fout is None:
        plt.show()
    else:
        plt.savefig(fout)
        # plt.show()
        print(f"save frame at t={t}")
    plt.close()


def plot_density(rho, t, para, figsize, fout=None):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    box = [0, para["Lx"], 0, para["Ly"]]

    if para["rho0"] <= 1.2:
        vmin, vmax = None, 6
    elif para["rho0"] >= 3:
        vmin, vmax = None, 10
    else:
        vmin, vmax = None, 6
    ax.imshow(rho, origin="lower", extent=box, vmin=vmin, vmax=vmax)
    if para["Ly"] > para["Lx"]:
        ax.yaxis.set_tick_params(rotation=90)

    if para["Lx"] == 1200:
        ax.set_xticks([0, 400, 800, 1200])
    title = r"$t=%g$" % (t * para["h"])
    plt.suptitle(title, fontsize="x-large")
    if fout is None:
        plt.show()
    else:
        plt.savefig(fout)
        # plt.show()
        print(f"save frame at t={t}")
    plt.close()


def plot_momentum(rho, vx, vy, t, para, figsize=(12, 7.5), fout=None):
    theta = np.arctan2(vy, vx)
    theta[theta < 0] += np.pi * 2
    theta *= 180 / np.pi
    module = np.sqrt(vx**2 + vy**2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    box = [0, para["Lx"], 0, para["Ly"]]
    if para["rho0"] >= 1.5:
        vmin, vmax = 0, 4
    elif para["rho0"] >= 1:
        vmin, vmax = 0, 4
    elif para["rho0"] > 0.55:
        vmin, vmax = 0, 3
    else:
        vmin, vmax = 0, 2
    RGB = map_v_to_rgb(theta, module, m_max=vmax)
    ax.imshow(RGB, extent=box, origin="lower")

    if para["Lx"] == para["Ly"]:
        title_one_row = False
    else:
        title_one_row = True
    if title_one_row:
        title_suffix = r"D=%g,\rho_0=%g,v_0=%g,{\rm seed}=%d,dt=%g, t=%g" % (
            para["D"], para["rho0"], para["v0"], para["seed"], para["h"],
            t * para["h"])
    else:
        title_suffix = r"D=%g,\rho_0=%g,v_0=%g" % (para["D"], para["rho0"],
                                                   para["v0"])
    if para["Lx"] == para["Ly"]:
        if not title_one_row:
            ax.set_title(r"${\rm seed}=%d, dt=%g,t=%g$" %
                         (para["seed"], para["h"], t * para["h"]),
                         fontsize="xx-large",
                         color="tab:red")
        plt.tight_layout(rect=[-0.02, -0.02, 1.02, 0.97])
        title = r"$L=%d,%s$" % (para["Lx"], title_suffix)
        plt.suptitle(title, y=0.995, fontsize="xx-large")
        fig.subplots_adjust(right=0.85)
        bbox = ax.get_position().get_points().flatten()
        # bbox = [xmin, ymin, xmax, ymax]
        bbox = ax.get_position().get_points().flatten()
        bbox[0] = 0.83
        bbox[2] = 0.08
        bbox[1] += 0.05
        bbox[3] = bbox[3] - bbox[1] - 0.1
        show_cb = True
    elif para["Lx"] == 1200 and para["Ly"] == 640:
        plt.tight_layout(rect=[-0.03, -0.06, 1.03, 0.96])
        title = r"$t=%g$" % (t * para["h"])
        plt.suptitle(title, y=0.999, fontsize="xx-large", wrap=True)
        show_cb = False
    elif para["Lx"] == 1200 and para["Ly"] == 320:
        plt.tight_layout(rect=[-0.06, -0.06, 1.00, 0.96])
        title = r"$t=%g$" % (t * para["h"])
        ax.set_xticks([0, 400, 800, 1200])
        ax.set_yticks([0, 160, 320])
        plt.suptitle(title, y=0.999, fontsize="x-large", wrap=True)
        show_cb = False
    elif para["Lx"] == 1200 and para["Ly"] == 160:
        plt.tight_layout(rect=[-0.06, -0.06, 1.00, 0.96])
        title = r"$t=%g$" % (t * para["h"])
        plt.suptitle(title, y=0.999, fontsize="x-large", wrap=True)
        show_cb = False
    elif para["Lx"] == 1200 and para["Ly"] == 1280:
        plt.tight_layout(rect=[-0.03, -0.03, 1.03, 0.98])
        title = r"$t=%g$" % (t * para["h"])
        plt.suptitle(title, y=0.999, fontsize="xx-large", wrap=True)
        show_cb = False
    elif para["Lx"] == 9600 and para["Ly"] == 1280:
        show_cb = False
        ax.set_ylabel(r"$t=%g$" % (t * para["h"]))
        ax.set_yticks([0, 400, 800, 1200])
        plt.tight_layout(rect=[-0.01, -0.04, 1.01, 1.04])
    elif para["Lx"] == para["Ly"] * 2:
        show_cb = False
        ax.set_ylabel(r"$t=%g$" % (t * para["h"]))
        plt.tight_layout(rect=[-0.02, -0.04, 1.02, 1.04])
    else:
        plt.tight_layout(rect=[-0.01, -0.05, 1.01, 0.96])
        title = r"$t=%g$" % (t * para["h"])
        plt.suptitle(title, y=0.999, fontsize="x-large", wrap=True)
        show_cb = False
    if show_cb:
        cb_ax = fig.add_axes(bbox)
        add_colorbar(cb_ax, vmin, vmax, 0, 360, "v")

    if fout is None:
        plt.show()
    else:
        plt.savefig(fout)
        print(f"same frame at t={t}")
    plt.close()


def plot_frames(f0, save_fig=True, fmt="jpg", which="momentum"):
    # para = get_para_field(f0)
    para = get_para(f0)
    prefix = "D:/data/wide_band_conti"
    # prefix = "imgs"
    if para["Lx"] == para["Ly"]:
        folder = "%s/%.1f_%g_%d_%.3f_%.3f_%d" % (prefix, para["v0"], para["h"],
                                                 para["Lx"], para["D"],
                                                 para["rho0"], para["seed"])
        if which == "both":
            figsize = (12, 7.5)
        else:
            figsize = (7, 6)
    else:
        folder = "%s/%.1f_%g_%d_%d_%.3f_%.3f_%d" % (
            prefix, para["v0"], para["h"], para["Lx"], para["Ly"], para["D"],
            para["rho0"], para["seed"])
        if which == "both":
            if para["Lx"] == 2400 and para["Ly"] == 5120:
                figsize = (6, 5.8)
            elif para["Lx"] == 1200 and para["Ly"] == 1280:
                figsize = (6, 3)
            if 3.5 < para["Lx"] / para["Ly"] <= 4:
                figsize = (12, 6)
            elif para["Lx"] // para["Ly"] == 2:
                figsize = (12, 10)
            elif para["Lx"] // para["Ly"] == 8:
                figsize = (12, 4)
            elif para["Lx"] // para["Ly"] == 12:
                figsize = (12, 3)
            elif para["Lx"] // para["Ly"] == 16:
                figsize = (16, 3)
            elif para["Lx"] // para["Ly"] == 24:
                figsize = (24, 3)
            elif para["Lx"] // para["Ly"] == 32:
                figsize = (32, 3)
            elif para["Ly"] // para["Lx"] == 2:
                figsize = (10, 8)
            elif para["Ly"] // para["Lx"] == 4:
                figsize = (4, 7.6)

        else:
            if para["Lx"] == 1200 and para["Ly"] == 640:
                figsize = (2.5, 1.5)
            elif para["Lx"] == 1200 and para["Ly"] == 320:
                figsize = (2.5, 1.0)
            elif para["Lx"] == 1200 and para["Ly"] == 160:
                figsize = (2.5, 0.7)
            elif para["Lx"] == 1200 and para["Ly"] == 1280:
                figsize = (2.5, 2.5)
            elif para["Lx"] == 1200 and para["Ly"] == 5120:
                figsize = (1.85, 5.8)
            elif para["Lx"] == 1200 and para["Ly"] == 10240:
                figsize = (1.85, 11.6)
            elif para["Lx"] == 2400 and para["Ly"] == 5120:
                figsize = (3, 5.8)
            elif para["Lx"] == 2400 and para["Ly"] == 9600:
                figsize = (2, 7)
            elif para["Lx"] == 2400 and para["Ly"] == 4800:
                figsize = (2, 3.8)
            elif para["Lx"] == 3600 and para["Ly"] == 5120:
                figsize = (4.0, 5.8)
            elif para["Lx"] == 9600 and para["Ly"] == 1280:
                figsize = (8, 1.3)
            elif para["Lx"] == 9600 and para["Ly"] == 600:
                figsize = (8, 1)
            elif para["Lx"] == 9600 and para["Ly"] == 300:
                figsize = (8, 0.75)
            elif para["Lx"] == 4800 and para["Ly"] == 300:
                figsize = (4, 0.75)
            elif para["Lx"] == 4800 and para["Ly"] == 600:
                figsize = (4, 1)
            elif para["Lx"] // para["Ly"] == 4:
                figsize = (2.5, 0.92)
            elif para["Lx"] // para["Ly"] == 8:
                figsize = (12, 2)
            elif para["Lx"] // para["Ly"] == 2:
                figsize = (5, 2.5)
                if para["Ly"] == 3600 and para["Lx"] == 9600:
                    figsize = (8, 3.2)
            elif para["Ly"] == 2 * para["Lx"]:
                figsize = (5, 6)
            else:
                figsize = (8, 4.2)
    if not os.path.exists(folder):
        os.mkdir(folder)
    existed_snap = glob.glob("%s/t=*.%s" % (folder, fmt))
    beg = len(existed_snap)
    frames = read_fields(f0, beg=beg)
    for i, (t, rho, vx, vy) in enumerate(frames):
        if save_fig:
            fout = "%s/t=%04d.%s" % (folder, beg + i, fmt)
        else:
            fout = None
        if which == "both":
            plot_density_momentum(rho, vx, vy, t, para, figsize, fout)
        elif which == "momentum":
            plot_momentum(rho, vx, vy, t, para, figsize, fout)
        elif which == "density":
            plot_density(rho, t, para, figsize, fout)


def plot_all_fields_series(pat="*_0.bin", fmt="jpg"):
    f0_list = glob.glob(f"fields/{pat}")
    for f0 in f0_list:
        print(f0)
        plot_frames(f0, True, fmt=fmt, which="momentum")


if __name__ == "__main__":
    pat = "2400_*_4.000_1.0_*_4_10000_0.1_*.bin"
    plot_all_fields_series(pat, fmt="jpg")
