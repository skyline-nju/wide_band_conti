import numpy as np
import struct
import os
import glob
import sys


def get_para(fin):
    para = {}
    s = os.path.basename(fin).rstrip(".bin").split("_")
    if len(s) == 9:
        para["Lx"] = int(s[0])
        para["Ly"] = para["Lx"]
        para["D"] = float(s[1])
        para["rho0"] = float(s[2])
        para["v0"] = float(s[3])
        para["seed"] = int(s[4])
        para["dx"] = int(s[5])
        para["dt"] = int(s[6])
        para["h"] = float(s[7])
        para["t_beg"] = int(s[8])
    elif len(s) == 10:
        para["Lx"] = int(s[0])
        para["Ly"] = int(s[1])
        para["D"] = float(s[2])
        para["rho0"] = float(s[3])
        para["v0"] = float(s[4])
        para["seed"] = int(s[5])
        para["dx"] = int(s[6])
        para["dt"] = int(s[7])
        para["h"] = float(s[8])
        para["t_beg"] = int(s[9])
    else:
        print("Error, failed to get paras from", fin)
        sys.exit(1)
    return para


def get_nframe(fin, match_all=False):
    def get_one_file_nframe(fin):
        with open(fin, "rb") as f:
            f.seek(0, 2)
            filesize = f.tell()
            para = get_para(fin)
            nx, ny = para["Lx"] // para["dx"], para["Ly"] // para["dx"]
            n = nx * ny
            frame_size = n * 12
            n_frame = filesize // frame_size
        return n_frame

    if match_all:
        para = get_para(fin)
        fin = get_files(para)
    if isinstance(fin, str):
        return get_one_file_nframe(fin)
    if isinstance(fin, list):
        n_frame = np.zeros(len(fin), int)
        t_beg = np.zeros_like(n_frame)
        dt = np.zeros_like(n_frame)
        for i, f in enumerate(fin):
            para = get_para(f)
            t_beg[i] = para["t_beg"]
            dt[i] = para["dt"]
            n_frame[i] = get_one_file_nframe(f)
        for i in range(n_frame.size - 1):
            n = (t_beg[i + 1] - t_beg[i]) // dt[i]
            if n > n_frame[i]:
                print("Warning,", n_frame[i], "frames in", fin[i],
                      "smaller than expected", n)
            elif n < n_frame[i]:
                print("Warning,", n_frame[i], "frames in", fin[i],
                      "larger than expected", n)
                n_frame[i] = n
        return n_frame
    else:
        print("Error, fin should be str or list of str for get_nframe")
        sys.exit(1)


def get_t_beg(fin):
    return get_para(fin)["t_beg"]


def get_files(para):
    pat_subfix = "{D:.3f}_{rho0:.3f}_{v0:.1f}_{seed}_{dx}_*_{h}_*.bin".format(
        **para)
    if para["Lx"] == para["Ly"]:
        pat = f"fields/{para['Lx']}_{pat_subfix}"
    else:
        pat = f"fields/{para['Lx']}_{para['Ly']}_{pat_subfix}"
    files = glob.glob(pat)
    files.sort(key=get_t_beg)
    return files


def read_one_file(fin, beg, end, sep, frame_idx0):
    with open(fin, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        para = get_para(fin)
        nx, ny = para["Lx"] // para["dx"], para["Ly"] // para["dx"]
        n = nx * ny
        framesize = n * 12
        nframe = filesize // framesize
        if nframe * framesize != filesize:
            print("Warning, filesize for", fin, "is", filesize,
                  ", but should be to be", nframe * framesize)
        f.seek(int(beg) * int(framesize))
        if end is None:
            file_end = filesize
        else:
            file_end = int(end) * int(framesize)
        frame_idx = frame_idx0
        while f.tell() < file_end:
            if frame_idx % sep == 0:
                buf = f.read(framesize)
                data = np.array(struct.unpack("%df" % (n * 3), buf))
                rho_m, vx_m, vy_m = data.reshape(
                    3, ny, nx) / (para["dx"] * para["dx"])
                t = para['t_beg'] + (beg + 1 + frame_idx -
                                     frame_idx0) * para["dt"]
                yield t, rho_m, vx_m, vy_m
            else:
                f.seek(framesize, 1)
            frame_idx += 1


def read_fields(fin, beg=0, end=None, sep=1, frame_idx0=1, single_file=False):
    if single_file:
        yield from read_one_file(fin, beg, end, sep, frame_idx0)
    else:
        para = get_para(fin)
        files = get_files(para)
        nframe = get_nframe(files)
        if beg < 0:
            beg += np.sum(nframe)
        print("%d/%d: %s" % (beg, np.sum(nframe), fin))

        beg_cur_file = 0
        end_cur_file = 0
        read_next = True
        for i, n in enumerate(nframe):
            beg_cur_file = end_cur_file
            end_cur_file += n
            if beg < end_cur_file:
                if end is None or end > end_cur_file:
                    yield from read_one_file(files[i], beg - beg_cur_file, n,
                                             sep, beg)
                else:
                    my_end = end - beg_cur_file
                    yield from read_one_file(files[i], beg - beg_cur_file,
                                             my_end, sep, beg)
                    read_next = False
                break
        if read_next:
            for j in range(i + 1, len(files)):
                beg_cur_file = end_cur_file
                end_cur_file += nframe[j]
                print(files[j])
                if end is None or end > end_cur_file:
                    yield from read_one_file(files[j], 0, nframe[j], sep, beg)
                else:
                    my_end = end - beg_cur_file
                    yield from read_one_file(files[j], 0, my_end, sep, beg)
                    break


def read_last_frame(fin):
    para = get_para(fin)
    files = get_files(para)
    f_last = files[-1]
    tot_frames = np.sum(get_nframe(files))
    last_file_frames = get_nframe(f_last)
    # print(tot_frames, last_file_frames)
    frames = read_one_file(f_last, last_file_frames - 1, None, 1,
                           tot_frames - 1)
    return next(frames)


if __name__ == "__main__":
    f0 = r"fields\1200_1280_0.300_1.600_1.0_4001_4_10000_0.1_0.bin"
    print(os.path.exists(f0))

    frames = read_fields(f0, single_file=False, beg=0, end=None)
    for i, frame in enumerate(frames):
        print(i)
