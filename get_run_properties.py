import os
import sys
import h5py
import numpy as np
from functions_import_essential import * # SG: imported essential functions


home_path = "/lfs/l7/hess/users/sghosh/projects/Satellite-trails/"
exec(
    open(
        home_path + "functions_import_essential.py", "r"
    ).read()
)
path_to_files = home_path + "projects/Satellite-trails/trail_properties/"
path_eval = "/lfs/l7/hess/users/sghosh/eval_high_data_utc_cut/high_pixel/"

nrun = int(sys.argv[1])

def read_single_run_utc_cut(nrun):
    runs_directory = "run"+str(nrun-nrun%200)+"-"+str(nrun-nrun%200+199)+"/"
    try:
        os.chdir(path_eval+runs_directory)
    except:
        print("folder /"+runs_directory+"/ does not exist")
        return
    try:
        print("Reading high_pix_"+str(nrun)+"_CT_5_cut.h5...")
        print(path_eval+runs_directory+str(nrun)+"/high_pix_"+str(int(nrun))+"_CT_5_cut.h5",)
        hfile = h5py.File(path_eval+runs_directory+str(nrun)+"/high_pix_"+str(int(nrun))+"_CT_5_cut.h5", "r")
        pix = np.array(hfile["Pix ID"]).astype(int)
        time = np.array(hfile["Time"])
        brightness = np.array(hfile["Brightness"])
        print("read brightness")
        new_high_pixel = [pix, time,brightness]
    except:
        print("Run number",nrun, "does not exist")
        os.chdir(home_path)
        new_high_pixel = np.array([[-1],[-1],[-1]])
        return new_high_pixel
    os.chdir(home_path)
    return new_high_pixel


# Old version:
# az_zen_nruns, az_zen_az, az_zen_zen, az_zen_utc, az_zen_time = read_az_zen_file([nrun])
# data = read_single_run_utc_cut(nrun)
# az_zen_time_in_day = convert_to_time_in_night(az_zen_time)

# New version:
hessall = h5py.File(home_path+"hessall.h5", "r")
hessall_nrun = np.array(hessall["run"])
hessall_zen = np.array(hessall["zenith"])
hessall_start = np.array(hessall["start_time"])
hessall.close()
index_nrun = np.where(hessall_nrun == nrun)[0]

if len(index_nrun) == 0:
    print(f"Run number {nrun} not found in hessall.h5!")
    sys.exit(1)

data = read_single_run_utc_cut(nrun)

if len(data[0])==1:
    print("no usable data in run "+str(nrun))
    hessall_time_in_day = convert_to_time_in_night(hessall_start[index_nrun][0])
    with open(path_to_files+str(int(nrun))+".txt", "w") as outfile:
        #outfile.write(str(nrun)+", "+str(az_zen_zen[0])+", "+ str(az_zen_az[0])+", "+
        #              str(az_zen_time_in_day[0])+", "+ str(-1)+", "+str(az_zen_time[0])+", "+
        #              str(-1)+ ", " + str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1))
        outfile.write(str(nrun)+", "+str(float(hessall_zen[index_nrun][0]))+", "+ str(-1)+", "+
                      str(hessall_time_in_day)+", "+ str(-1)+", "+str(int(hessall_start[index_nrun][0]))+", "+
                      str(-1)+ ", " + str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1))
        outfile.write('\n')
    exit()
print("Run "+ str(int(nrun))+": "+str(len(data[0]))+ " entries")
track, possible_meteorite = track_sorter(data[0],data[1],data[2],typed_nn_pix)
# possible_meteorite, trash = adding_possible_meteorites(possible_meteorite)
track = angle_adding_of_tracks(track)

if track[0][0][0] !=-1:
    print("Number of tracks: "+ str(len(track)))
else:
    print("Number of tracks: 0")
    hessall_time_in_day = convert_to_time_in_night(hessall_start[index_nrun][0])
    with open(path_to_files+str(int(nrun))+".txt", "w") as outfile:
        outfile.write(str(nrun)+", "+str(float(hessall_zen[index_nrun][0]))+", "+ str(-1)+", "+ 
                      str(hessall_time_in_day)+", "+ str(-1)+", "+str(int(hessall_start[index_nrun]))+", "+
                      str(-1)+ ", " + str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1)+", "+str(-1))
        outfile.write('\n')
    exit()

velocities = []
brightness = []
start_time = []
stop_time = []
duration_on_cam = []
unique_pix_in_track = []

with open(path_to_files+str(int(nrun))+".txt", "w") as outfile:
    for i in range(len(track)):
        if track[i][0][0] == -1:
            continue
        if i == 0:
            print("Tracks:")
        if len(np.unique(np.array(track[i][1])))>1:
            v = get_velocity_in_deg_per_s(np.array(track[i][0]), np.array(track[i][1]))
        else:
            v = 100000
        b = round(get_mean_brightness(np.array(track[i][2])),1)
        b_std = round(np.std(np.array(track[i][2])),1)
        cb = get_cumul_brightness(np.array(track[i][2]))
        duration_on_cam.append(track[i][1][-1]-track[i][1][0])
        unique_pix_in_track.append(len(np.unique(track[i][0])))
        velocities.append(v)
        brightness.append(b)
        start_time.append(track[i][1][0])
        stop_time.append(track[i][1][-1])
        hessall_time_in_day = convert_to_time_in_night([track[i][1][0]])

        print(str(nrun)+", "+str(float(hessall_zen[index_nrun][0]))+", "+ str(-1)+", ")
        print(str(hessall_time_in_day[0])+", "+ str(i)+", "+str(int(hessall_start[index_nrun][0]))+", ")
        print(str(round(track[i][1][0],1))+ ", " + str(round(track[i][1][-1]-track[i][1][0],1))+", ")
        print(str(len(np.unique(track[i][0])))+", ")
        print(str(-1)+", "+str(round(v,1))+", "+str(round(b,1))+", ")
        print(str(round(np.max(track[i][2]),1)))

        outfile.write(str(nrun)+", "+str(float(hessall_zen[index_nrun][0]))+", "+ str(-1)+", "+
                      str(hessall_time_in_day[0])+", "+ str(i)+", "+str(int(hessall_start[index_nrun]))+", "+
                      str(round(track[i][1][0],1))+ ", " + str(round(track[i][1][-1]-track[i][1][0],1))+", "+
                      str(len(np.unique(track[i][0])))+", "+
                      str(-1)+", "+str(round(v,3))+", "+str(round(b,1))+", "+
                      str(round(np.max(track[i][2]),1)))
        outfile.write('\n')
outfile.close()
