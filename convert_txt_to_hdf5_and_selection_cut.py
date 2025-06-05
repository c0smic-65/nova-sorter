import sys
import os
import numpy as np
import h5py
from astropy.time import Time

# def single_selection_cut(data_high_pixel):
#     max_counts = 100        # SG: This removes pix IDs that appear more than 100 times, WHY?
#     values, counts = np.unique(data_high_pixel[0], return_counts=True)
#     allowed_pix_ids = values[counts < max_counts] # SG: git rid of it!
#     mask = np.isin(data_high_pixel[0], allowed_pix_ids)
#     new_pix = np.array(data_high_pixel[0])[mask]
#     new_time = np.array(data_high_pixel[1])[mask]
#     new_brightness = np.array(data_high_pixel[2])[mask]
#     high_pixel_cut = np.array([new_pix, new_time, new_brightness])
#     return high_pixel_cut

def single_selection_cut(data_high_pixel):
    new_pix = np.array(data_high_pixel[0])
    new_time = np.array(data_high_pixel[1])
    new_brightness = np.array(data_high_pixel[2])
    high_pixel_cut = np.array([new_pix, new_time, new_brightness])
    return high_pixel_cut


path_high_pixel = "/lfs/l7/hess/users/sghosh/eval_high_data_utc/high_pixel"
path_high_pixel_cut = "/lfs/l7/hess/users/sghosh/eval_high_data_utc_cut/high_pixel"
if len(sys.argv) < 2:
    print("need number as entry")
    exit()

nrun = int(sys.argv[1])
print(f"Convert {nrun}...")
subpath_to_run = f"/run{int(nrun - nrun % 200)}-{int(nrun - nrun % 200 + 199)}/{int(nrun)}/"
path_to_run = path_high_pixel + subpath_to_run
path_to_run_cut = path_high_pixel_cut + subpath_to_run

try:
    txt_files = os.listdir(path_to_run)
    print(txt_files)
except Exception as e:
    print(f"os.listdir does not work with this run: {e}")
    exit()
if len(txt_files) == 0:
    print("No Dataset found for this run")
    exit()

if f"high_pix_{nrun}_CT_5.h5" in txt_files:
    print(f"high_pix_{nrun}_CT_5.h5 already exists!")
    # os.remove(path_to_run + f"high_pix_{nrun}_CT_5.txt")  # <-- This line is commented out, so .txt is kept
    exit()

try:
    pix = np.loadtxt(path_to_run + f"high_pix_{int(nrun)}_CT_5.txt",
                     usecols=0, delimiter=";", ndmin=1)
    brightness = np.loadtxt(path_to_run + f"high_pix_{int(nrun)}_CT_5.txt",
                            usecols=1, delimiter=";", ndmin=1)
    time = np.loadtxt(path_to_run + f"high_pix_{int(nrun)}_CT_5.txt",
                      usecols=2, delimiter=";", dtype=str, ndmin=1)
except Exception as e:
    print(f"Error loading txt data for run {nrun}: {e}")
    exit()

# Confirm time formatting
times = [item.replace("UTC: ", "") for item in time]
t = Time(times, format="iso", scale="utc")
t.format = "unix"
t = t.value

# os.remove(path_to_run + f"high_pix_{nrun}_CT_5.txt")  # <-- This is commented out, keeping the .txt file

data = [pix, t, brightness]
data = single_selection_cut(data)

if len(data[0]) == 0:
    data = [[-1], [-1], [-1]]

outdir = path_to_run_cut
if not os.path.exists(outdir):
    os.makedirs(outdir)
hfile = h5py.File(outdir + f"high_pix_{nrun}_CT_5_cut.h5", "w")
hfile.create_dataset("Pix ID", data=data[0], compression="gzip", compression_opts=9)
hfile.create_dataset("Time", data=data[1], compression="gzip", compression_opts=9)
hfile.create_dataset("Brightness", data=data[2], compression="gzip", compression_opts=9)
hfile.close()