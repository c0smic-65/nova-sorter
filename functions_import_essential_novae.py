########################################################
#This is a collection of all functions used during the 
#developement of the trail finding.
#Also by executing this the variables nn_pix and typed_nn_pix 
#are set, defining the neighbourhood pixel IDs of each of the 
#1764 pixels as a nested List and a numba.typed.List() respectively.
#
#
#The main functions to read the data and find the tracks are:
#read_single_run_utc_cut(nrun)
#track_sorter_fast(data)
#
#To display them the following functions are used: 
#plot_time_in_track(track)
#plot_brightness_in_track(track)
#
#This collection of functions still contains some other functions
#that were used during the developement of the algorithm.
#

########################################################

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tables import *
import os
import operator
import sys
import csv
import h5py
import time
from astropy.time import Time
import astropy.units as u
from numba import jit,njit,int32,float32
from numba.typed import List
from scipy.ndimage import median_filter

home_path = "/lfs/l7/hess/users/sghosh/"

#
path_eval = os.path.join(os.path.join(home_path,"eval_high_data_utc"),"high_pixel")
path_eval_cut = os.path.join(os.path.join(home_path,"eval_high_data_utc_cut"),"high_pixel")
print("path_eval is set to: \""+path_eval+"\"")
print("path_eval_cut is set to: \""+path_eval_cut+"\"")


#Get camera geometry from file
flash_geom_id = np.loadtxt(os.path.join(home_path,"flashcam_geometry.txt"), usecols = 0, delimiter=";").astype(int)
flash_geom_x = np.loadtxt(os.path.join(home_path,"flashcam_geometry.txt"), usecols = 1, delimiter=";")
flash_geom_y = np.loadtxt(os.path.join(home_path,"flashcam_geometry.txt"), usecols = 2, delimiter=";")
flash_geom_x = np.array(flash_geom_x)
flash_geom_y = np.array(flash_geom_y)
a = np.around(np.sort(np.unique(flash_geom_x)),5)
b = np.around(np.sort(np.unique(flash_geom_y)),5)
sort_x = np.around(np.sort(np.unique(flash_geom_x)),5)
sort_y = np.around(np.sort(np.unique(flash_geom_y)),5)
min_x_dif = a[1]-a[0]
min_y_dif = b[1]-b[0]

#Produces list of list of neighbourhood pixels for each pixel
def get_nn_pix(size):
    nn_pix = []
    for pix in flash_geom_id:
        mask_x = np.logical_and(flash_geom_x<min_x_dif*size*0.6+flash_geom_x[pix],
                                flash_geom_x>-min_x_dif*size*0.6+flash_geom_x[pix])
        mask_xy = np.logical_and(flash_geom_y[mask_x]<min_y_dif*(size-0.1)+flash_geom_y[pix],
                                 flash_geom_y[mask_x]>-min_y_dif*size+flash_geom_y[pix])
        nn_pix.append(flash_geom_id[mask_x][mask_xy])
    return nn_pix

nn_pix = get_nn_pix(10)

#Same as above, but using numba.typed.List() for nested lists
def get_typed_nn_pix(size):
    nn_pix = List()
    for pix in flash_geom_id:
        mask_x = np.logical_and(flash_geom_x<min_x_dif*size*0.6+flash_geom_x[pix],
                                flash_geom_x>-min_x_dif*size*0.6+flash_geom_x[pix])
        mask_xy = np.logical_and(flash_geom_y[mask_x]<min_y_dif*(size-0.1)+flash_geom_y[pix],
                                 flash_geom_y[mask_x]>-min_y_dif*size+flash_geom_y[pix])
        nn_pix.append(List(flash_geom_id[mask_x][mask_xy]))
    return nn_pix

typed_nn_pix = get_typed_nn_pix(10)


#Produce arrays of x and y coordinates for array of pixel IDs
@jit
def pix_to_xy(trail_pix):
    x = np.array([flash_geom_x[int(trail_pix[0])]])
    y = np.array([flash_geom_y[int(trail_pix[0])]])
    for i in range(1,len(trail_pix)):
        x = np.concatenate((x, np.array([flash_geom_x[int(trail_pix[i])]])))
        y = np.concatenate((y, np.array([flash_geom_y[int(trail_pix[i])]])))    
    return x, y

#Simple function for a scatter plot
def plot_scatter(x, y, z):
    #fig= plt.figure(figsize=(6,5), )
    plt.scatter(x,y,c=z, cmap = "jet")
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)
    plt.xlabel("Camera x-pos")
    plt.ylabel("Camera y-pos")
    plt.title("Run {}".format(nrun)+", CT {}".format(telId))
    #cbar = plt.colorbar()
    #cbar.set_label("Time  in run [s]")
    
    
#Base function for the following two
def plot_from_pix(pix, z):
    x = []
    y = []
    for i in range(len(pix)):
        x.append(flash_geom_x[int(pix[i])])
        y.append(flash_geom_y[int(pix[i])])
    plt.scatter(x,y, c = z, cmap = "jet")
    plt.xlim(-1.25,1.25)    
    plt.ylim(-1.25,1.25)  
    plt.xlabel("Camera x-pos")
    plt.ylabel("Camera y-pos")
    return

#
def plot_brightness_from_pix(Data):
    fig = plt.figure(figsize = (8,6))
    #Data should be in form of [pix, time, brightness]
    unique_pix = np.unique(Data[0])
    avg_brightness = np.zeros(len(unique_pix))
    for i in range(len(unique_pix)):
        avg_brightness[i] = np.average(Data[2][np.where(Data[0]==unique_pix[i])[0]])
    plot_from_pix(unique_pix, avg_brightness)
    draw_box()
    cb = plt.colorbar()
    cb.set_label("Average Brightnes [MHz]")
    return

def plot_time_from_pix(Data):
    #Data should be in form of [pix, time, brightness] 
    fig = plt.figure(figsize = (8,6))
    plot_from_pix(Data[0], Data[1])
    draw_box()
    cbar = plt.colorbar()
    cbar.set_label("Time in run [s]")
    return

#taking all outer pixel positions and draw the hexagon
def draw_box():
    mask_x_top = np.around(np.array(flash_geom_x), 5) == np.max(sort_x)
    mask_x_bot = np.around(np.array(flash_geom_x), 5) == np.min(sort_x)
    mask_y_top = np.around(np.array(flash_geom_y), 5) == np.max(sort_y)
    mask_y_bot = np.around(np.array(flash_geom_y), 5) == np.min(sort_y)
    
    y_secondtolast = sort_y[-2]
    y_secondtofirst = sort_y[1]
    mask1 = np.around(np.array(flash_geom_y), 6) == y_secondtolast
    mask2 = np.around(np.array(flash_geom_y), 6) == y_secondtofirst
    
    box_x = [flash_geom_x[mask_x_top][0],
             np.max(np.around(flash_geom_x[mask1], 6)),
            np.max(flash_geom_x[mask_y_top]),
            np.min(flash_geom_x[mask_y_top]),
            flash_geom_x[mask_x_bot][0],
            flash_geom_x[mask_x_bot][-1],
            np.min(flash_geom_x[mask_y_bot]),
            np.max(flash_geom_x[mask_y_bot]),
             np.max(np.around(flash_geom_x[mask2], 6)),
            flash_geom_x[mask_x_top][0],
            flash_geom_x[mask_x_top][0]]
    box_y = [flash_geom_y[mask_x_top][0],
             y_secondtolast,
             flash_geom_y[mask_y_top][0],
             flash_geom_y[mask_y_top][-1],
             np.max(flash_geom_y[mask_x_bot]),
             np.min(flash_geom_y[mask_x_bot]),
             flash_geom_y[mask_y_bot][0],
             flash_geom_y[mask_y_bot][-1],
             y_secondtofirst,
             flash_geom_y[mask_x_top][0],
             flash_geom_y[mask_x_top][0]]
    for i in range(len(box_x)-2):
        if (box_x[i+1]-box_x[i]) !=0:
            if (box_x[i+1]-box_x[i])> 0:
                x_vals = np.arange(box_x[i], box_x[i+1], 0.001)
            else:
                x_vals = np.arange(box_x[i+1], box_x[i], 0.001)
            m_box = (box_y[i+1]-box_y[i])/(box_x[i+1]-box_x[i])
            t_box = box_y[i]- m_box*box_x[i]
            y_vals = m_box*x_vals + t_box
            plt.plot(x_vals, y_vals, c = "black")
        else:
            x_vals = [box_x[i]]
            if box_y[i]<box_y[i+1]:
                y_vals = np.arange(box_y[i], box_y[i+1], 0.001)
            else:
                y_vals = np.arange(box_y[i+1], box_y[i], 0.001)
            for j in range(len(y_vals)-1):
                x_vals.append(x_vals[0])
            plt.vlines(box_x[i], box_y[i], box_y[i+1], colors ="black")
    return 


#older version used to read in files before the 900MHz cut was applied
def read_single_run(nrun):
    runs_directory = "run"+str(nrun-nrun%200)+"-"+str(nrun-nrun%200+199)
    path_eval_runs = path_eval+runs_directory
    try:
        os.chdir(path_eval_runs)
    except:
        print("folder /"+runs_directory+"/ does not exist")
    try:   
        print(os.getcwd())
        print("Reading high_pix_"+str(nrun)+"_CT_5.h5...")
        hfile = h5py.File(str(nrun)+"/high_pix_"+str(int(nrun))+"_CT_5.h5", "r")
        pix = np.array(hfile["Pix ID"]).astype(int)
        time = np.array(hfile["Time"])
        brightness = np.array(hfile["Brightness"])
        new_high_pixel = [pix, time,brightness]
    except:
        print("Run number",nrun, "does not exist")
        os.chdir(home_path)
        new_high_pixel = np.array([[-1],[-1],[-1]])
        return new_high_pixel
    os.chdir(home_path)
    return new_high_pixel

#get starting time of run, which is subtracted from unix times to get time in run
#return 0 if stating time is not found
def get_start_time(nrun):
    if type(nrun) != type(1):
        print("use int entry")
        return 0
    try:
        hfile = h5py.File(home_path+"projects/Satellite-trails/run_properties.h5", "r")
    except:
        print("File \"run_properties.h5\" not found")
        return 0
    nruns = np.array(hfile["nrun"])
    if nrun not in nruns:
        print(nrun, "not in \"run_properties.h5\"")
        return 0
    index_nrun = np.where(nruns==nrun)[0][0]
    t0 = np.array(hfile["t0"])[index_nrun]
    #t0 is time of run start in unix
    hfile.close()
    return t0

#Use this to read in data of a specific run number
#Please ignore the utc in the naming, the time in the files is 
#actually in unix time.
#The time in the output gives the time in the run in seconds.
def read_single_run_utc_cut(nrun):
    runs_directory = "/"+"run"+str(nrun-nrun%200)+"-"+str(nrun-nrun%200+199)+"/"
    try:
        os.chdir(path_eval_cut+runs_directory)
    except:
        print("folder /"+runs_directory+"/ does not exist")
    t0 = get_start_time(nrun)
    try:   
        print("Reading high_pix_"+str(nrun)+"_CT_5_cut.h5...")
        print(path_eval_cut+runs_directory+str(nrun)+"/high_pix_"+str(int(nrun))+"_CT_5_cut.h5",)
        hfile = h5py.File(path_eval_cut+runs_directory+str(nrun)+"/high_pix_"+str(int(nrun))+"_CT_5_cut.h5", "r")
        print("read pix:")
        pix = np.array(hfile["Pix ID"]).astype(int)
        print("read time:")
        time = np.around(np.array(hfile["Time"])-t0, 1)
        brightness = np.array(hfile["Brightness"])
        print("read brightness:")
        new_high_pixel = [pix, time,brightness]
    except:
        print("Run number",nrun, "does not exist")
        os.chdir(home_path)
        new_high_pixel = np.array([[-1],[-1],[-1]])
        return new_high_pixel
    os.chdir(home_path)
    return new_high_pixel



def apply_selection_cut(dict_high_pixel, azzen_time):
    max_counts = 70
    high_pixel_cut = {}
    k = 0
    for nrun in dict_high_pixel.keys():
        print(nrun)
        high_pixel_cut[nrun] = {}
        values, counts = np.unique(dict_high_pixel[nrun]["pix"], return_counts=True)
        mask_max_counts = counts<max_counts       
        st = set(values[mask_max_counts])
        result = [i for i, e in enumerate(dict_high_pixel[nrun]["pix"]) if e in st]
        high_pixel_cut[nrun]["pix"] = np.array(dict_high_pixel[nrun]["pix"])[result].astype(int)
        high_pixel_cut[nrun]["brightness"] = np.array(dict_high_pixel[nrun]["brightness"])[result]
        high_pixel_cut[nrun]["time"] = np.array(dict_high_pixel[nrun]["time"])[result]-azzen_time[k]
        k+=1
    return high_pixel_cut

def single_selection_cut(data_high_pixel, azzen_time):
    max_counts = 100
    values, counts = np.unique(data_high_pixel[0], return_counts=True)
    mask_max_counts = counts<max_counts  
    st = set(values[mask_max_counts])
    result = [i for i, e in enumerate(data_high_pixel[0]) if e in st]
    new_pix = np.array(data_high_pixel[0])[result]
    new_time = np.array(data_high_pixel[1])[result]-azzen_time[0]
    new_brightness = np.array(data_high_pixel[2])[result]
    high_pixel_cut = np.array([new_pix, new_time, new_brightness])
    return high_pixel_cut

def single_selection_cut_no_azzentime(data_high_pixel):
    #max_counts = 100
    values, counts = np.unique(data_high_pixel[0], return_counts=True)
    mask_max_counts = counts<max_counts
    st = set(values[mask_max_counts])
    result = [i for i, e in enumerate(data_high_pixel[0]) if e in st]
    new_pix = np.array(data_high_pixel[0])[result]
    new_time = np.array(data_high_pixel[1])[result]
    new_brightness = np.array(data_high_pixel[2])[result]
    high_pixel_cut = np.array([new_pix, new_time, new_brightness])
    return high_pixel_cut



def plot_track_time(data):
    if len(data[0]) == 1:
        print("Invalid data, plot is not done")
        return
    x = flash_geom_x[np.array(data[0]).astype(int)]
    y = flash_geom_y[np.array(data[0]).astype(int)]
    plt.scatter(x,y, c = data[1], cmap = "jet", vmin = np.min(data[1])-1, vmax = np.max(data[1])+1 )
    plt.colorbar(label = "time in run [s]")    
    plt.xlabel("Camera x-pos")
    plt.ylabel("Camera y-pos")
    draw_box()
    plt.show()
    return


# Plot all tracks' last entry of a pixel timestamp with
# the colorbar denoting the time sonce the beginning of each track
def plot_time_in_track(tracks):
    if len(tracks[0][0]) == 1:
        print("Invalid data, plot is not done")
        return
    data = [np.array([]),np.array([]),np.array([])]
    for N in range(len(tracks)):
        data[1] = np.append(data[1], np.array(tracks[N][1])-tracks[N][1][0])
        data[0] = np.append(data[0], np.array(tracks[N][0]))
        data[2] = np.append(data[2], np.array(tracks[N][2]))
    x = flash_geom_x[np.array(data[0]).astype(int)]
    y = flash_geom_y[np.array(data[0]).astype(int)]
    plt.figure(figsize = (6,4), constrained_layout = True)
    plt.scatter(x,y, c = data[1], cmap = "jet", 
                vmin = np.min(data[1])-1, vmax = np.max(data[1])+1,
                marker = "H", s = 30)
    plt.colorbar(label = "Time in Track [s]")
    plt.xlabel("X-Coordinate [m]")#, fontsize = 14)
    plt.ylabel("Y-Coordinate [m]")#, fontsize = 14)
    draw_box()
    return

# Plot the average brightness in a pixel of a track
# for a numba.typed.List() of tracks
def plot_brightness_in_track(tracks):
    if tracks[0][0][0] == -1:
        print("Invalid data, plot is not done")
        return
    Data = [np.array([]),np.array([]),np.array([])]
    for N in range(len(tracks)):
        Data[1] = np.append(Data[1], np.array(tracks[N][1])-tracks[N][1][0])
        Data[0] = np.append(Data[0], np.array(tracks[N][0]))
        Data[2] = np.append(Data[2], np.array(tracks[N][2]))
    fig = plt.figure(figsize = (7* 0.69,5* 0.69), constrained_layout = True)
    ax = plt.subplot(111)
    x = []
    y = []
    brightness = []
    unique_pix = np.unique(Data[0])
    for i in range(len(unique_pix)):
        x.append(flash_geom_x[int(unique_pix[i])])
        y.append(flash_geom_y[int(unique_pix[i])])
        b = [Data[2][j] for j in range(len(Data[0])) if int(Data[0][j])== int(unique_pix[i])]
        brightness.append(np.average(b))
    print(len(x), len(y), len(brightness))
    im = ax.scatter(x,y, c = brightness, cmap = "jet",
                    marker = "H", s = 13)
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    fig.colorbar(im, label = "Average Brightness [MHz]")
    draw_box()
    return 

#Use track_sorter_fast(data), as it is less tedious to write than this one
#This produces the list of trails and a list of all the rest
#The rest also contains meteorites, hence the naming of the variable
@jit
def track_sorter(dict_pix, dict_tmp, dict_brightness, typed_nn_pixels):
    # typed_nn_pixels inst needed to work together with @jit
    if len(dict_pix) == 0:
        tracks = List([[[-1.], [-1.], [-1.]]])
        possible_meteorites = List([[[-1.], [-1.], [-1.]]])
        # No valid tracks if pix list empty
        return tracks, possible_meteorites

    pix = list(dict_pix)
    tmp = list(dict_tmp)
    brightness = list(dict_brightness)
    nn_pixels = list(typed_nn_pixels)
    # Initialize first track
    tracks = List([[[float(pix[0])], [float(tmp[0])], [float(brightness[0])]]])

    # Cluster hits into tracks using spatial adjacency & time window
    for i in range(1, len(pix)):
        appended_to_track = False
        for N in range(len(tracks)):
            # Widened clustering window: 60 s for slow-evolving sources
            if tmp[i] - 60 < tracks[N][1][-1]:
                # Check if current pixel neighbors recent hits
                if len([x for x in nn_pixels[int(pix[i])] if x in tracks[N][0][-20:]]) > 0:
                    tracks[N][0].append(pix[i])
                    tracks[N][1].append(tmp[i])
                    tracks[N][2].append(brightness[i])
                    appended_to_track = True
                    break
        if not appended_to_track:
            # Start a new track when no adjacency found
            tracks.append([[float(pix[i])], [float(tmp[i])], [float(brightness[i])]])

    possible_meteorites = List([[[-1.], [-1.], [-1.]]])
    to_remove = np.full(len(tracks), -1)

    # Apply filtering cuts for star/nova candidates
    for N in range(len(tracks)):
        unique_pix = np.unique(np.array(tracks[N][0]))
        unique_t = np.unique(np.array(tracks[N][1]))

        # 1) Spatial-extent cut: no more than 3 unique pixels
        if len(unique_pix) > 10:
            to_remove[N] = N
            continue

        # 2) Temporal-sampling cut: average gap <= 5 s for slow curves
        if len(unique_t) > 1:
            time_diff = unique_t[1:] - unique_t[:-1]
            if np.average(time_diff) > 5.0:
                to_remove[N] = N
                continue
        else:
            # Insufficient time points: drop track
            to_remove[N] = N
            continue

        # 3) Large-trail guard: reject tracks spanning >40 pixels
        if len(unique_pix) > 40:
            to_remove[N] = N
            continue

        # 4) Velocity cut: retain near-stationary (<= 0.001 deg/s)
        velo = get_velocity(tracks[N][0], tracks[N][1])
        if velo > 0.002:
            to_remove[N] = N
            continue

    # Remove marked tracks in reverse order to preserve indices
    idxs = to_remove[to_remove != -1]
    for idx in sorted(idxs, reverse=True):
        del tracks[int(idx)]

    # Ensure non-empty output
    if len(tracks) == 0:
        tracks = List([[[-1.], [-1.], [-1.]]])

    return tracks, possible_meteorites




#Old Idea, for meteorite searches, however it was scrapped
@jit 
def adding_possible_meteorites(pos_met):
    tmp = List(pos_met)
    trash = List()
    if (len(pos_met)==0 or len(tmp)==0):
        trash.append([[-1.],[-1.],[-1.]])
        return tmp, trash
    index_to_del = List()
    for N in range(len(tmp)-1): #Add all trails that are at the same time or just 1s apart from each other
        if len([x for x in tmp[-(N+1)][1] if x in tmp[-(N+2)][1]])>0:
            tmp[-(N+2)][0] = tmp[-(N+2)][0]+ tmp[-(N+1)][0]#pix
            tmp[-(N+2)][1] = tmp[-(N+2)][1]+ tmp[-(N+1)][1]#tmp
            tmp[-(N+2)][2] = tmp[-(N+2)][2]+ tmp[-(N+1)][2]#brightness
            index_to_del.append(-(N+1)+len(tmp))
    for N in range(len(tmp)): #put everything in trash less than 4 unique pixels or longer than 100s(possible tracks?)
        if (len(np.unique(np.array(tmp[N][0])))<6 or len(np.unique(np.array(tmp[N][1])))>100):
            trash.append(tmp[N])
            if N in index_to_del:
                continue
            index_to_del.append(N)
            continue
        for i in np.unique(np.array(tmp[N][0])):#reject
            a = np.array(tmp[N][0])
            count = (a == i).sum()
            if count>8:
                if N in index_to_del:
                    continue
                index_to_del.append(N)
                continue
    index_to_del = sorted(index_to_del)
    for i in range(len(index_to_del)):
        del tmp[index_to_del[-(i+1)]]
    return tmp, trash


#Idea of adding together potentially severed tracks be the angle of the vector
#between the first and last entry, however needs more work to properly function
@jit
def angle_adding_of_tracks(tracklist):
    track_added = []
    thetas = []
    if tracklist[0][0][0]== -1:
        track_added.append(tracklist[0])
        thetas.append(-1.)
        return track_added
    max_angle_dif = 5.
    for i in range(len(tracklist)):
        x,y = pix_to_xy(tracklist[i][0])
        x_min_time = x[np.where(np.array(tracklist[i][1]) == np.min(np.array(tracklist[i][1])))]
        y_min_time = y[np.where(np.array(tracklist[i][1]) == np.min(np.array(tracklist[i][1])))]
        x_max_time = x[np.where(np.array(tracklist[i][1]) == np.max(np.array(tracklist[i][1])))]
        y_max_time = y[np.where(np.array(tracklist[i][1]) == np.max(np.array(tracklist[i][1])))]
        v = np.array([np.average(x_max_time)-np.average(x_min_time), np.average(y_max_time)-np.average(y_min_time)] )
        if np.sqrt(v.dot(v)) == 0:
            track_added.append(tracklist[0])
            thetas.append(-1.-max_angle_dif)
            continue
        #vector v is calculated between the avg of pixel coordinates of first and latest times respectively
        theta = np.arccos( v[0]/np.sqrt(v.dot(v)) )*180/np.pi
        if i >0:
            if (theta-max_angle_dif<thetas[-1] and theta+max_angle_dif>thetas[-1]):
                max_time_dif = 30.
                if track_added[-1][1][-1]<tracklist[i][1][0]-max_time_dif:
                    #don't add together if max_time_dif is too large 
                    track_added.append(tracklist[i])
                    thetas.append(theta)
                    continue
                track_added[-1][0] = track_added[-1][0]+tracklist[i][0]
                track_added[-1][1] = track_added[-1][1]+tracklist[i][1]
                track_added[-1][2] = track_added[-1][2]+tracklist[i][2]
                thetas.append(theta)
                continue
        track_added.append(tracklist[i])
        thetas.append(theta)
    return track_added


#used in the track_sorter
@jit
def get_velocity(trail_pix, trail_time):
    # Ensure that there are at least two time points and the times are not the same
    if len(trail_time) < 2 or trail_time[-1] == trail_time[0]:
        return 0.0  # Return zero velocity if times are identical or there's not enough data
    
    x_trail, y_trail = pix_to_xy(trail_pix)
    d = np.hypot(x_trail[-1] - x_trail[0], y_trail[-1] - y_trail[0])
    t = trail_time[-1] - trail_time[0]
    
    # Avoid division by zero by checking if t is greater than a small epsilon
    if t == 0:
        return 0.0  # Avoid division by zero if the time difference is zero
    
    return d / t



@jit
def get_velocity_in_deg_per_s(trail_pix, trail_time):
    #Values should be for Flashcam, but could not find them yet, maybe the same as first camera?
    #FOV over maximum x-vals would be 3.8deg
    deg_per_m = 0.067/0.042
    x_trail, y_trail = pix_to_xy(trail_pix)
    d = np.sqrt((x_trail[-1]-x_trail[0])**2+(y_trail[-1]-y_trail[0])**2)
    t = trail_time[-1]-trail_time[0]
    velo = d/t* deg_per_m
    return velo


@jit
def get_mean_brightness(trail_brightness):
    mean_bright = np.mean(trail_brightness)
    return mean_bright
@jit
def get_cumul_brightness(trail_brightness):
    cumul_bright = np.sum(trail_brightness)
    return cumul_bright


def convert_to_time_in_night(azzen_time):
    t = Time(azzen_time, format = 'unix', scale='utc')
    t.format = "iso"
    a = Time(t, format = "iso",out_subfmt = "date_hms")
    b = t.to_value("iso",subfmt = "date")
    b = Time(b, format = "iso", scale = "utc")
    azzen_time_in_day = np.array((a-b).value*3600*24).astype(int)
    return azzen_time_in_day

def increase_threshold(high_pix):
    values, counts = np.unique(high_pix[0], return_counts=True)
    i = 0
    mask = high_pix[0]>-1
    while len(values)>400:
        mask = high_pix[2]>900+i*50
        values, counts = np.unique(high_pix[0][mask], return_counts=True)
        #print(len(values), 900+i*50)
        i+=1
    if i>0:
        print("New threshold at "+str(int(900+i*50)))
    else:
        print("No new threshold")
    high_pix_cut = []
    high_pix_cut.append(high_pix[0][mask])
    high_pix_cut.append(high_pix[1][mask])
    high_pix_cut.append(high_pix[2][mask])
    high_pix_cut = np.array(high_pix_cut)
    return high_pix_cut

# SG: Function to plot the light curve of the brightest track

def plot_lightcurve_brightest_track(tracks,
                                    metric: str = "cumulative",
                                    ax=None):
    """
    Identify the brightest track returned by `track_sorter_fast`
    and draw its brightness-vs-time curve without any de-duplication.

    Parameters
    ----------
    tracks : list / numba.typed.List
        Output of `track_sorter_fast`.
    metric : {'cumulative', 'mean'}, optional
        How to rank tracks:
        * 'cumulative' – largest Σ(brightness)  (default)
        * 'mean'       – highest ⟨brightness⟩
    ax : matplotlib.axes.Axes, optional
        Draw on this axes; otherwise create a new figure.

    Returns
    -------
    idx        : int       – index of the selected track
    times      : ndarray   – original time array (shifted so t₀ = 0)
    brightness : ndarray   – brightness array for that track
    """
    if not tracks or tracks[0][0][0] == -1:
        raise ValueError("No valid tracks supplied")

    # --- convert to NumPy so reductions work cleanly
    track_arrs = [(np.asarray(tr[1], float),   # times
                   np.asarray(tr[2], float))   # brightness
                  for tr in tracks]

    # --- select track
    scores = [(b.mean() if metric == "mean" else b.sum())
              for (_, b) in track_arrs]
    idx = int(np.argmax(scores))
    times, brightness = track_arrs[idx]
    times = times - times[0]                  # start at zero

    # --- plot
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    ax.plot(times, brightness, marker="o", lw=1.2, ms=3,
            label=f"Track {idx}")
    ax.set_xlabel("Time in track [s]")
    ax.set_ylabel("Pixel brightness")
    desc = "cumulative brightness" if metric == "cumulative" else "mean brightness"
    ax.set_title(f"Light-curve of brightest track ({desc})")
    ax.legend()

    return idx, times, brightness

# SG: Function to plot the light curve of the brightest track
# but only keeping the peak brightness at each time stamp.

def plot_lightcurve_brightest_track_peak(tracks,
                                         metric: str = "cumulative",
                                         ax=None):
    """
    Same idea as above, but if several samples share the same time stamp
    (because the streak spans multiple pixels in one exposure), keep only
    the single brightest value – a robust “peak-pixel” light-curve.

    Parameters & Returns
    --------------------
    Identical to `plot_lightcurve_brightest_track`.
    """
    if not tracks or tracks[0][0][0] == -1:
        raise ValueError("No valid tracks supplied")

    track_arrs = [(np.asarray(tr[1], float),   # times
                   np.asarray(tr[2], float))   # brightness
                  for tr in tracks]

    scores = [(b.mean() if metric == "mean" else b.sum())
              for (_, b) in track_arrs]
    idx = int(np.argmax(scores))
    times, brightness = track_arrs[idx]
    times = times - times[0]

    # --- collapse duplicate times by taking the maximum brightness
    t_unique, inv = np.unique(times, return_inverse=True)
    b_peak = np.full_like(t_unique, -np.inf, dtype=float)
    np.maximum.at(b_peak, inv, brightness)

    # --- plot
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    ax.plot(t_unique, b_peak, marker="o", lw=1.2, ms=3,
            label=f"Track {idx} (peak)")
    ax.set_xlabel("Time in track [s]")
    ax.set_ylabel("Peak pixel brightness")
    desc = "cumulative brightness" if metric == "cumulative" else "mean brightness"
    ax.set_title(f"Light-curve of brightest track ({desc})")
    ax.legend()

    return idx, t_unique, b_peak



def plot_total_flux_light_curve(tracks, track_number=0):
    """
    Plots the total-sum light curve for a given track from the track sorter output.
    
    Parameters:
    -----------
    tracks : list
        Output from the track sorter function. Each element should be a track represented as
        [pix_ids, times, brightnesses], where:
          - pix_ids: list of pixel IDs (ints or floats)
          - times: list of timestamps (ints or floats) corresponding to each pixel measurement
          - brightnesses: list of brightness values (floats) corresponding to each pixel measurement
    track_number : int, optional
        Index of the track to plot (default is 0, i.e., the first track).
    
    The function groups all brightness measurements at each unique time, sums them to get the total flux,
    and then plots total flux vs. time.
    """
    # Validate track_number
    if track_number < 0 or track_number >= len(tracks):
        raise ValueError(f"track_number {track_number} is out of range for the provided tracks.")

    # Unpack the specified track
    pix_ids, times, brightnesses = tracks[track_number]
    
    # Ensure times and brightnesses have the same length
    if len(times) != len(brightnesses):
        raise ValueError("Length mismatch: 'times' and 'brightnesses' must be the same length.")
    
    # Identify unique timestamps and sort them
    unique_times = sorted(set(times))
    
    # Compute total flux at each timestamp
    total_flux = []
    for t in unique_times:
        # Collect brightnesses at time t
        b_at_t = [b for tt, b in zip(times, brightnesses) if tt == t]
        total_flux.append(sum(b_at_t))
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(unique_times, total_flux, linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Total Flux")
    plt.title(f"Light Curve for Track {track_number}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage (commented out; replace with actual 'tracks' data when available):
# tracks = [
#     # Example track: [[pix_ids...], [times...], [brightnesses...]]
#     ([1, 2, 3, 1, 2], [0.0, 0.0, 0.0, 1.0, 1.0], [10.0, 5.0, 2.0, 8.0, 3.0]),
#     # Additional tracks...
# ]
# plot_total_flux_light_curve(tracks)

def plot_mean_flux_light_curve(tracks, track_number=0):
    """
    Plots the mean-flux light curve for a given track from the track sorter output.
    
    Parameters:
    -----------
    tracks : list
        Output from the track sorter function. Each element should be a track represented as
        [pix_ids, times, brightnesses], where:
          - pix_ids: list of pixel IDs (ints or floats)
          - times: list of timestamps (ints or floats) corresponding to each pixel measurement
          - brightnesses: list of brightness values (floats) corresponding to each pixel measurement
    track_number : int, optional
        Index of the track to plot (default is 0, i.e., the first track).
    
    The function groups all brightness measurements at each unique time, computes their mean
    to get the mean flux per time, and then plots mean flux vs. time.
    """
    # Validate track_number
    if track_number < 0 or track_number >= len(tracks):
        raise ValueError(f"track_number {track_number} is out of range for the provided tracks.")

    # Unpack the specified track
    pix_ids, times, brightnesses = tracks[track_number]
    
    # Ensure times and brightnesses have the same length
    if len(times) != len(brightnesses):
        raise ValueError("Length mismatch: 'times' and 'brightnesses' must be the same length.")
    
    # Identify unique timestamps and sort them
    unique_times = sorted(set(times))
    
    # Compute mean flux at each timestamp
    mean_flux = []
    for t in unique_times:
        # Collect brightnesses at time t
        b_at_t = [b for tt, b in zip(times, brightnesses) if tt == t]
        mean_flux.append(np.mean(b_at_t))
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(unique_times, mean_flux, linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Mean Flux")
    plt.title(f"Mean-Flux Light Curve for Track {track_number}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage (commented out; replace with actual 'tracks' data when available):
# tracks = [
#     # Example track: [[pix_ids...], [times...], [brightnesses...]]
#     ([1, 2, 3, 1, 2], [0.0, 0.0, 0.0, 1.0, 1.0], [10.0, 5.0, 2.0, 8.0, 3.0]),
#     # Additional tracks...
# ]
# plot_mean_flux_light_curve(tracks)

def plot_binned_flux_light_curve(tracks, track_number=0, time_bin_size=0.5):
    """
    Plots the mean-flux light curve for a given track, binned into intervals of width `time_bin_size`.
    
    Parameters
    ----------
    tracks : list
        Output from the track sorter function. Each element should be a track represented as
        [pix_ids, times, brightnesses].
    track_number : int, optional
        Index of the track to plot (default is 0).
    time_bin_size : float, optional
        Width of the time‐bins (in the same units as `times`), e.g. 0.5 for half‐second bins.
    """
    # Validate inputs
    if not (0 <= track_number < len(tracks)):
        raise ValueError(f"track_number {track_number} is out of range.")
    pix_ids, times, brightnesses = tracks[track_number]
    if len(times) != len(brightnesses):
        raise ValueError("Length mismatch: 'times' and 'brightnesses' must match.")
    
    times = np.array(times)
    brightnesses = np.array(brightnesses)
    
    # Define bin edges from min to max time
    t_min, t_max = times.min(), times.max()
    bins = np.arange(t_min, t_max + time_bin_size, time_bin_size)
    
    # Digitize times into bins: each time falls into bin index i where bins[i-1] <= t < bins[i]
    bin_indices = np.digitize(times, bins)
    
    # Compute mean flux in each bin
    binned_time_centers = []
    binned_flux = []
    for i in range(1, len(bins)):
        in_bin = (bin_indices == i)
        if not np.any(in_bin):
            # skip empty bins or append NaN
            continue
        binned_time_centers.append((bins[i-1] + bins[i]) / 2)
        binned_flux.append(brightnesses[in_bin].mean())
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(binned_time_centers, binned_flux, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Mean Flux")
    plt.title(f"Binned Mean-Flux Light Curve (Δt = {time_bin_size}s) for Track {track_number}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_median_flux_light_curve(tracks, track_number=0):
    """
    Plots the median-flux light curve for a given track from the track sorter output.
    
    Parameters:
    -----------
    tracks : list
        Output from the track sorter function. Each element should be a track represented as
        [pix_ids, times, brightnesses], where:
          - pix_ids: list of pixel IDs (ints or floats)
          - times: list of timestamps (ints or floats) corresponding to each pixel measurement
          - brightnesses: list of brightness values (floats) corresponding to each pixel measurement
    track_number : int, optional
        Index of the track to plot (default is 0, i.e., the first track).
    
    The function groups all brightness measurements at each unique time, computes their median
    to get the median flux per time, and then plots median flux vs. time.
    """
    # Validate track_number
    if track_number < 0 or track_number >= len(tracks):
        raise ValueError(f"track_number {track_number} is out of range for the provided tracks.")

    # Unpack the specified track
    pix_ids, times, brightnesses = tracks[track_number]
    
    # Ensure times and brightnesses have the same length
    if len(times) != len(brightnesses):
        raise ValueError("Length mismatch: 'times' and 'brightnesses' must be the same length.")
    
    # Identify unique timestamps and sort them
    unique_times = sorted(set(times))
    
    # Compute median flux at each timestamp
    median_flux = []
    for t in unique_times:
        # Collect brightnesses at time t
        b_at_t = [b for tt, b in zip(times, brightnesses) if tt == t]
        median_flux.append(np.median(b_at_t))
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(unique_times, median_flux, linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Median Flux")
    plt.title(f"Median-Flux Light Curve for Track {track_number}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage (commented out; replace with actual 'tracks' data when available):
# tracks = [
#     # Example track: [[pix_ids...], [times...], [brightnesses...]]
#     ([1, 2, 3, 1, 2], [0.0, 0.0, 0.0, 1.0, 1.0], [10.0, 5.0, 2.0, 8.0, 3.0]),
#     # Additional tracks...
# ]
# plot_median_flux_light_curve(tracks)

