import numpy as np
# import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    fin = "fields/1200_640_0.300_3.400_1.0_2001_2_10000_0.1_0.bin"
    get_y_averaged_fields(fin)
