import numpy as np
import matplotlib.pyplot as plt
# import glob
import os
from read_fields import get_para, read_fields, get_nframe


def read_y_averaged_fields(fin=None,
                           Lx=None,
                           Ly=None,
                           D=None,
                           rho0=None,
                           v0=1.0,
                           seed=None):
    if fin is None:
        if Ly is None:
            Ly = Lx
        fin = f"space_time/{Lx}_{Ly}_{D:.3f}_{rho0:.3f}_{v0:.1f}_{seed}.npz"
    with np.load(fin) as data:
        t = data['t']
        x = data['x']
        rho_x = data['rho_x']
        mx_x = data['mx_x']
        my_x = data['my_x']
    return t, x, rho_x, mx_x, my_x


def read_x_averaged_fields(fin=None,
                           Lx=None,
                           Ly=None,
                           D=None,
                           rho0=None,
                           v0=1.0,
                           seed=None):
    if fin is None:
        if Ly is None:
            Ly = Lx
        fin = f"transverse_profiles/{Lx}_{Ly}_" \
            f"{D:.3f}_{rho0:.3f}_{v0:.1f}_{seed}.npz"
    with np.load(fin) as data:
        t = data['t']
        y = data['y']
        rho_y = data['rho_y']
        mx_y = data['mx_y']
        my_y = data['my_y']
    return t, y, rho_y, mx_y, my_y


def get_y_averaged_fields(fin):
    para = get_para(fin)
    fnpz = f"space_time/{para['Lx']}_{para['Ly']}_{para['D']:.3f}_" \
        f"{para['rho0']:.3f}_{para['v0']:.1f}_{para['seed']}.npz"

    nframes = np.sum(get_nframe(fin, True))
    nx = para['Lx'] // para['dx']
    x = np.arange(nx) * para['dx'] + para['dx'] / 2
    t_arr = np.zeros(nframes)
    rho_x, mx_x, my_x = np.zeros((3, nframes, nx), dtype=np.float32)

    if os.path.exists(fnpz):
        t0, x0, rho_x0, mx_x0, my_x0 = read_y_averaged_fields(fnpz)
        x = x0
        beg = t0.size
        t_arr[:beg] = t0
        rho_x[:beg], mx_x[:beg], my_x[:beg] = rho_x0, mx_x0, my_x0
    else:
        beg = 0

    if beg < nframes:
        frames = read_fields(fin, beg=beg)
        for i, (t, rho, vx, vy) in enumerate(frames):
            j = i + beg
            t_arr[j] = i * para['h']
            rho_x[j] = np.mean(rho, axis=0)
            mx_x[j] = np.mean(vx, axis=0)
            my_x[j] = np.mean(vy, axis=0)
        np.savez_compressed(fnpz,
                            t=t_arr,
                            x=x,
                            rho_x=rho_x,
                            mx_x=mx_x,
                            my_x=my_x)
    return t, x, rho_x, mx_x, my_x


def get_x_averaged_fields(fin):
    para = get_para(fin)
    fnpz = f"transverse_profiles/{para['Lx']}_{para['Ly']}_{para['D']:.3f}_" \
        f"{para['rho0']:.3f}_{para['v0']:.1f}_{para['seed']}.npz"

    nframes = np.sum(get_nframe(fin, True))
    ny = para['Ly'] // para['dx']
    y = np.arange(ny) * para['dx'] + para['dx'] / 2
    t_arr = np.zeros(nframes)
    rho_y, mx_y, my_y = np.zeros((3, nframes, ny), dtype=np.float32)

    if os.path.exists(fnpz):
        t0, y0, rho_y0, mx_y0, my_y0 = read_x_averaged_fields(fnpz)
        y = y0
        beg = t0.size
        t_arr[:beg] = t0
        rho_y[:beg], mx_y[:beg], my_y[:beg] = rho_y0, mx_y0, my_y0
    else:
        beg = 0

    if beg < nframes:
        frames = read_fields(fin, beg=beg)
        for i, (t, rho, vx, vy) in enumerate(frames):
            j = i + beg
            t_arr[j] = i * para['h']
            rho_y[j] = np.mean(rho, axis=1)
            mx_y[j] = np.mean(vx, axis=1)
            my_y[j] = np.mean(vy, axis=1)
        np.savez_compressed(fnpz,
                            t=t_arr,
                            y=y,
                            rho_y=rho_y,
                            mx_y=mx_y,
                            my_y=my_y)
    return t, y, rho_y, mx_y, my_y


if __name__ == "__main__":
    # fin = "fields/1200_640_0.300_3.400_1.0_2001_2_10000_0.1_0.bin"
    # get_y_averaged_fields(fin)

    # fin = "fields/1200_10240_0.300_1.600_1.0_1164001_4_10000_0.1_0.bin"
    # get_x_averaged_fields(fin)

    fin = "transverse_profiles/1200_10240_0.300_1.600_1.0_1164001.npz"
    t, y, rho_y, mx_y, my_y = read_x_averaged_fields(fin)
    plt.imshow(rho_y-1.6, origin="lower", cmap="bwr", aspect="auto")
    plt.show()
    plt.close()
