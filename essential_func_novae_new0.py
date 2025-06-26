#%%
from functions_import_essential_novae import *
#%%
typed_nn_pix = get_typed_nn_pix(5)
print(typed_nn_pix)

new_high_pixel = read_single_run_utc_cut(161133)
pix = new_high_pixel[0]
tmp = new_high_pixel[1]
brightness = new_high_pixel[2]

tracks, possible_meteorites = track_sorter(pix, tmp, brightness, typed_nn_pix)

print(tracks)

for i, track in enumerate(tracks):
    pix = track[0]
    times = track[1]
    brightness = track[2]
    print(f"Track {i} — Unique Pixels: {len(set(pix))}")
    print(f"  First  → Pixel ID: {pix[0]}, Time: {times[0]}, Brightness: {brightness[0]}")
    print(f"  Last   → Pixel ID: {pix[-1]}, Time: {times[-1]}, Brightness: {brightness[-1]}")

plot_brightness_in_track(tracks)
plot_binned_mean_flux_light_curve(tracks, track_number=0, time_bin_size=50)
plt.show()
#%%

typed_nn_pix = get_typed_nn_pix(5)
print(typed_nn_pix)

new_high_pixel = read_single_run_utc_cut(161096)
pix = new_high_pixel[0]
tmp = new_high_pixel[1]
brightness = new_high_pixel[2]

tracks, possible_meteorites = track_sorter(pix, tmp, brightness, typed_nn_pix)

print(tracks)

for i, track in enumerate(tracks):
    pix = track[0]
    times = track[1]
    brightness = track[2]
    print(f"Track {i} — Unique Pixels: {len(set(pix))}")
    print(f"  First  → Pixel ID: {pix[0]}, Time: {times[0]}, Brightness: {brightness[0]}")
    print(f"  Last   → Pixel ID: {pix[-1]}, Time: {times[-1]}, Brightness: {brightness[-1]}")

plot_brightness_in_track(tracks)
plt.show()

#%%

plot_total_flux_light_curve(tracks, track_number=0)
plot_mean_flux_light_curve(tracks, track_number=0)
plot_median_flux_light_curve(tracks, track_number=0)
#%%

typed_nn_pix = get_typed_nn_pix(5)
print(typed_nn_pix)

new_high_pixel = read_single_run_utc_cut(161097)
pix = new_high_pixel[0]
tmp = new_high_pixel[1]
brightness = new_high_pixel[2]

tracks, possible_meteorites = track_sorter(pix, tmp, brightness, typed_nn_pix)

print(tracks)

for i, track in enumerate(tracks):
    pix = track[0]
    times = track[1]
    brightness = track[2]
    print(f"Track {i} — Unique Pixels: {len(set(pix))}")
    print(f"  First  → Pixel ID: {pix[0]}, Time: {times[0]}, Brightness: {brightness[0]}")
    print(f"  Last   → Pixel ID: {pix[-1]}, Time: {times[-1]}, Brightness: {brightness[-1]}")

plot_brightness_in_track(tracks)
plt.show()

#%%

plot_total_flux_light_curve(tracks, track_number=0)
plot_mean_flux_light_curve(tracks, track_number=0)
plot_median_flux_light_curve(tracks, track_number=0)
#%%

plot_lightcurve_brightest_track_peak(tracks, metric="mean")
plt.show()

#%%
plt.plot(tracks[0][1], tracks[0][2])
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# assume tracks[0][1] is your time array, tracks[0][2] your pixel brightnesses
times      = np.array(tracks[0][1])
brightness = np.array(tracks[0][2])

# find the unique times, and an inverse index for grouping
unique_times, inv_idx = np.unique(times, return_inverse=True)

# sum up all brightness values that share the same time
total_flux = np.bincount(inv_idx, weights=brightness)

plt.plot(unique_times, total_flux, '-o', markersize=2)
plt.xlabel('Time [s]')
plt.ylabel('Total Flux [arb. units]')
plt.title('Lightcurve: summed pixel brightness per frame')
plt.show()
#%%
import numpy as np, matplotlib.pyplot as plt

# flatten into numpy arrays
t = np.array(tracks[0][1])
b = np.array(tracks[0][2])

# group by time
unique_t, inv = np.unique(t, return_inverse=True)

# total flux & pixel counts per timestamp
flux_sum  = np.bincount(inv, weights=b)
npix      = np.bincount(inv)

# mean flux per pixel
flux_mean = flux_sum / npix

plt.plot(unique_t, flux_mean)
plt.xlabel('Time [s]')
plt.ylabel('Mean Pixel Brightness')
plt.title('Lightcurve: mean pixel brightness / frame')
plt.show()

#%%
plot_mean_flux_light_curve(tracks, track_number=0)
#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# load the columns, skipping “#” lines
data = np.loadtxt('/lfs/l7/hess/users/sghosh/tracks_161097.txt', 
                  delimiter='\t',
                  comments='#',
                  usecols=(0,1,2,3))

# unpack
track_ids, pixel_ids, times, brightness = data.T

# mask just track 0
mask0 = track_ids == 0
times0 = times[mask0]

# count occurrences of each time
cnt = Counter(times0)

# sort by time
t_sorted = np.array(sorted(cnt.keys()))
n_pix    = np.array([cnt[t] for t in t_sorted])

# plot
plt.figure(figsize=(8,4))
plt.plot(t_sorted, n_pix*1000, marker='.', linestyle='--')
plt.plot(unique_t, flux_mean, color='red' )
plt.xlabel('Time [s]')
plt.ylabel('Number of pixels triggered')
plt.title('Pixels per frame vs Time')
plt.tight_layout()
plt.grid()
plt.legend(['Pixels per frame', 'Mean Pixel Brightness'])
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 2D Gaussian plus constant background:
def gauss2d(XY, A, x0, y0, sigma_x, sigma_y, B):
    x, y = XY
    ex = np.exp(-((x - x0)**2/(2*sigma_x**2)
                  + (y - y0)**2/(2*sigma_y**2)))
    return (A * ex + B)
#%%
# from flashcam_geometry.txt
coord_map = {pid: (x, y)
             for pid, x, y
             in zip(flash_geom_id, flash_geom_x, flash_geom_y)}
#%%
def unpack_frame(frame_pids, frame_fluxes, coord_map):
    """Given arrays of pixel IDs & brightnesses, 
       return x, y, z arrays for fitting."""
    coords = np.array([coord_map[pid] for pid in frame_pids])
    x, y = coords[:,0], coords[:,1]
    z    = np.array(frame_fluxes)
    return x, y, z
#%%
def fit_frame_psf(frame_pids, frame_fluxes, coord_map,
                  min_pixels=10):
    # unpack coords & flux
    x, y, z = unpack_frame(frame_pids, frame_fluxes, coord_map)

    # if too few pixels, bail out
    if len(z) < min_pixels:
        return np.array([np.nan]*7)  # A, x0, y0, sx, sy, B, Ftot

    # ...then do the curve_fit exactly as before...
    popt, _ = curve_fit(gauss2d, np.vstack((x,y)), z, p0=initial_guess)
    A, x0, y0, sx, sy, B = popt
    Ftot = 2*np.pi*A*sx*sy
    return np.array([A, x0, y0, sx, sy, B, Ftot])

#%%
# before your loop, build arrays and index map:
geom_ids = flash_geom_id           # e.g. [0,1,2,3,…]
x_geom   = flash_geom_x            # same order
y_geom   = flash_geom_y
id2idx   = {pid:i for i,pid in enumerate(geom_ids)}

def fit_frame_psf_full(frame_pids, frame_fluxes):
    # 1) Build full-brightness array
    z_full = np.zeros_like(x_geom)
    for pid, val in zip(frame_pids, frame_fluxes):
        z_full[id2idx[pid]] = val

    # 2) Now unpack x,y from the full geometry
    x = x_geom
    y = y_geom
    z = z_full

    # 3) Initial parameter guesses
    A0    = z.max() - np.median(z)
    x0_0  = (x*z).sum()/z.sum()
    y0_0  = (y*z).sum()/z.sum()
    s0    = np.std(z)      # very rough
    B0    = np.median(z)
    p0    = [A0, x0_0, y0_0, s0, s0, B0]

    # 4) Fit the model
    popt, _ = curve_fit(gauss2d, np.vstack((x,y)), z, p0=p0)

    # 5) Extract and compute total flux
    A_fit, x0_fit, y0_fit, sx_fit, sy_fit, B_fit = popt
    F_tot = 2*np.pi*A_fit*sx_fit*sy_fit

    return np.array([A_fit, x0_fit, y0_fit, sx_fit, sy_fit, B_fit, F_tot])
#%%

# assume flash_geom_id, flash_geom_x, flash_geom_y are already defined
geom_ids = flash_geom_id               # array of all pixel IDs
x_geom   = flash_geom_x                # same order
y_geom   = flash_geom_y
# map pixel ID → position in these arrays
id2idx   = {pid: idx for idx, pid in enumerate(geom_ids)}
n_pix    = len(geom_ids)
#%%
import numpy as np
from scipy.optimize import curve_fit

# model: 2D Gaussian + constant
def gauss2d(XY, A, x0, y0, sigma_x, sigma_y, B):
    x, y = XY
    return A * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))) + B

def fit_frame_full(frame_pids, frame_fluxes):
    """
    frame_pids/fluxes : only those pixels that fired
    returns array [A, x0, y0, sx, sy, B, F_tot]
    """
    # 1) build full-flux vector (zeros for unfired pixels)
    z = np.zeros(n_pix, dtype=float)
    for pid, f in zip(frame_pids, frame_fluxes):
        z[id2idx[pid]] = f

    # 2) pack full coords
    x = x_geom
    y = y_geom

    # 3) initial guesses
    B0    = np.median(z)
    A0    = z.max() - B0
    x0_0  = (x*z).sum() / z.sum()  # centroid
    y0_0  = (y*z).sum() / z.sum()
    s0    = np.std(z)              # rough
    p0    = [A0, x0_0, y0_0, s0, s0, B0]

    # 4) fit
    popt, _ = curve_fit(gauss2d, np.vstack((x,y)), z, p0=p0)
    A, x0, y0, sx, sy, B = popt

    # 5) compute total PSF flux
    F_tot = 2 * np.pi * A * sx * sy
    return np.array([A, x0, y0, sx, sy, B, F_tot])
#%%
# flatten your track data
all_pids   = np.array(tracks[0][0], dtype=int)
all_times  = np.array(tracks[0][1])
all_fluxes = np.array(tracks[0][2])

# group by time
unique_times, inv_idx = np.unique(all_times, return_inverse=True)

# prepare result array
# columns: A, x0, y0, sx, sy, B, F_tot
psf_results = np.zeros((len(unique_times), 7))

for i, t in enumerate(unique_times):
    mask         = (inv_idx == i)
    frame_pids   = all_pids[mask]
    frame_fluxes = all_fluxes[mask]
    psf_results[i] = fit_frame_full(frame_pids, frame_fluxes)
#%%
geom_ids = flash_geom_id
x_geom   = flash_geom_x
y_geom   = flash_geom_y
id2idx   = {pid: i for i, pid in enumerate(geom_ids)}
min_x_dif = np.min(np.diff(np.sort(np.unique(x_geom))))
min_y_dif = np.min(np.diff(np.sort(np.unique(y_geom))))
n_pix    = len(geom_ids)
#%%
def gauss2d(XY, A, x0, y0, sx, sy, B):
    x, y = XY
    return A * np.exp(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2))) + B
#%%
all_pids   = np.array(tracks[0][0], int)
all_times  = np.array(tracks[0][1])
all_fluxes = np.array(tracks[0][2])
unique_times, inv_idx = np.unique(all_times, return_inverse=True)
#%%
psf_results = np.zeros((len(unique_times), 7))
for i, t in enumerate(unique_times):
    mask         = (inv_idx == i)
    frame_pids   = all_pids[mask]
    frame_fluxes = all_fluxes[mask]
    psf_results[i] = fit_frame_full(frame_pids, frame_fluxes)
#%%
from scipy.optimize import curve_fit

def fit_frame_full(frame_pids, frame_fluxes):
    # Assemble a full-length brightness array (zeros for unfired)
    z = np.zeros(n_pix, float)
    for pid, val in zip(frame_pids, frame_fluxes):
        z[id2idx[pid]] = val

    # Coordinates always the full camera
    x = x_geom
    y = y_geom

    # Initial guesses
    B0   = np.median(z)
    A0   = z.max() - B0
    x0_0 = (x*z).sum() / z.sum()
    y0_0 = (y*z).sum() / z.sum()
    s0   = np.sqrt(min_x_dif * min_y_dif)
    p0   = [A0, x0_0, y0_0, s0, s0, B0]

    # Bounds to keep the fit reasonable
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    lower = [    0, x_min, y_min, min_x_dif, min_y_dif,    0  ]
    upper = [5*A0, x_max, y_max, x_max-x_min, y_max-y_min, 5*B0]

    try:
        popt, _ = curve_fit(
            gauss2d,
            np.vstack((x, y)),
            z,
            p0=p0,
            bounds=(lower, upper),
            maxfev=5000
        )
    except Exception:
        # if it still fails, return NaNs so your loop won’t crash
        return np.full(7, np.nan)

    A, x0, y0, sx, sy, B = popt
    F_tot = 2 * np.pi * A * sx * sy
    return np.array([A, x0, y0, sx, sy, B, F_tot])
#%%
# flatten your selected track (e.g. tracks[0])
all_pids   = np.array(tracks[0][0], dtype=int)
all_times  = np.array(tracks[0][1])
all_fluxes = np.array(tracks[0][2])

# get unique timestamps and an inverse index
unique_times, inv_idx = np.unique(all_times, return_inverse=True)
#%%
# preallocate: columns = [A, x0, y0, sx, sy, B, F_tot]
psf_results = np.zeros((len(unique_times), 7))

for i in range(len(unique_times)):
    mask             = (inv_idx == i)
    frame_pids       = all_pids[mask]
    frame_fluxes     = all_fluxes[mask]
    psf_results[i]   = fit_frame_full(frame_pids, frame_fluxes)
#%%
import matplotlib.pyplot as plt

F_tot = psf_results[:, -1]  # last column
plt.plot(unique_times, F_tot, '-o', ms=3)
plt.xlabel('Time [s]')
plt.ylabel('PSF‐fit Total Flux')
plt.title('Gaussian PSF Photometry Lightcurve')
plt.show()
#%%
all_pids   = np.array(tracks[0][0], dtype=int)
all_times  = np.array(tracks[0][1],      dtype=float)
all_fluxes = np.array(tracks[0][2],      dtype=float)

unique_times, inv_idx = np.unique(all_times, return_inverse=True)

print("Range of raw times:", all_times.min(), "→", all_times.max())
print("First 10 unique_times:", unique_times[:10])
#%%
# psf_results shape = (N_frames, 7)
F_tot = psf_results[:, -1]

print("PSF‐fit fluxes:  min =", np.nanmin(F_tot),
      " max =", np.nanmax(F_tot),
      " any NaNs? ", np.isnan(F_tot).any())
#%%
# Build full z-array (zeros + fired)
z = np.zeros(n_pix, float)
for pid, val in zip(frame_pids, frame_fluxes):
    z[id2idx[pid]] = val

# Estimate background as the median of *all* pixels
B0 = np.median(z)
# Subtract it
z_sub = z - B0  
#%%
def gauss2d_circ(XY, A, x0, y0, s):
    x, y = XY
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2*s**2))
#%%
from scipy.optimize import curve_fit

def fit_frame_simple(frame_pids, frame_fluxes):
    # 1) Build full-flux array z, then subtract median background
    z = np.zeros(n_pix, float)
    for pid, val in zip(frame_pids, frame_fluxes):
        z[id2idx[pid]] = val
    B0    = np.median(z)
    z_sub = z - B0

    # 2) Pack full coords
    x = x_geom
    y = y_geom

    # 3) Initial guesses:
    A0   = z_sub.max()
    x0_0 = (x * z_sub).sum() / z_sub.sum()
    y0_0 = (y * z_sub).sum() / z_sub.sum()
    s0   = np.sqrt(min_x_dif * min_y_dif)  # ~pixel scale
    p0   = [A0, x0_0, y0_0, s0]

    # 4) Reasonable bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    lower = [    0,  x_min,    y_min,    min(min_x_dif,min_y_dif)]
    upper = [5*A0, x_max,    y_max,    max(x_max-x_min,y_max-y_min)]

    # 5) Fit only A, x0, y0, s
    try:
        popt, _ = curve_fit(
            gauss2d_circ,
            np.vstack((x, y)),
            z_sub,
            p0=p0,
            bounds=(lower, upper),
            maxfev=5000
        )
    except Exception:
        return np.full(5, np.nan)  # A, x0, y0, s, B0

    A_fit, x0_fit, y0_fit, s_fit = popt
    # total flux under the Gaussian surface:
    F_psf = 2 * np.pi * A_fit * s_fit**2
    return np.array([A_fit, x0_fit, y0_fit, s_fit, B0, F_psf])
#%%
# flatten your track
all_pids   = np.array(tracks[0][0], dtype=int)
all_times  = np.array(tracks[0][1],      dtype=float)
all_fluxes = np.array(tracks[0][2],      dtype=float)

unique_times, inv_idx = np.unique(all_times, return_inverse=True)

# prepare result: 6 columns [A, x0, y0, s, B0, F_psf]
res = np.zeros((len(unique_times), 6))

for i in range(len(unique_times)):
    mask        = (inv_idx == i)
    frame_pids   = all_pids[mask]
    frame_fluxes = all_fluxes[mask]
    res[i]       = fit_frame_simple(frame_pids, frame_fluxes)
#%%
import matplotlib.pyplot as plt

F_psf = res[:, -1]  # last column
good  = ~np.isnan(F_psf)

plt.figure(figsize=(8,4))
plt.plot(unique_times[good], F_psf[good], '-o', ms=3)
plt.xlabel('Time [s]')
plt.ylabel('Gaussian PSF Total Flux')
plt.title('Circular-Gaussian PSF Photometry Lightcurve')
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


# Build lookup & pixel‐spacing
geom_ids  = flash_geom_id
x_geom    = flash_geom_x
y_geom    = flash_geom_y
n_pix     = len(geom_ids)
id2idx    = {pid: i for i, pid in enumerate(geom_ids)}
min_x_dif = np.min(np.diff(np.sort(np.unique(x_geom))))
min_y_dif = np.min(np.diff(np.sort(np.unique(y_geom))))

# 2) Circular 2D Gaussian model (no explicit background term)
def gauss2d_circ(XY, A, x0, y0, s):
    x, y = XY
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * s**2))

# 3) Single‐frame fitter
def fit_frame_simple(frame_pids, frame_fluxes):
    # build full‐camera flux array
    z = np.zeros(n_pix, float)
    for pid, f in zip(frame_pids, frame_fluxes):
        z[id2idx[pid]] = f
    # subtract median background
    B0    = np.median(z)
    z_sub = z - B0

    # pack coords
    x = x_geom
    y = y_geom

    # initial guesses
    A0    = np.max(z_sub)
    x0_0  = (x * z_sub).sum() / z_sub.sum()
    y0_0  = (y * z_sub).sum() / z_sub.sum()
    s0    = np.sqrt(min_x_dif * min_y_dif)
    p0    = [A0, x0_0, y0_0, s0]

    # bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    lower = [    0, x_min,    y_min,    min(min_x_dif, min_y_dif)]
    upper = [5*A0, x_max,    y_max,    max(x_max - x_min, y_max - y_min)]

    # fit
    try:
        popt, _ = curve_fit(
            gauss2d_circ,
            np.vstack((x, y)),
            z_sub,
            p0=p0,
            bounds=(lower, upper),
            maxfev=5000
        )
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, B0, np.nan])

    A_fit, x0_fit, y0_fit, s_fit = popt
    # total PSF flux
    F_psf = 2 * np.pi * A_fit * s_fit**2
    return np.array([A_fit, x0_fit, y0_fit, s_fit, B0, F_psf])

# 4) Extract & group your track data (using track 0 as example)
#    Replace this with whichever track you want
pix_vals   = np.array(tracks[0][0], dtype=int)
time_vals  = np.array(tracks[0][1],      dtype=float)
flux_vals  = np.array(tracks[0][2],      dtype=float)

unique_times, inv_idx = np.unique(time_vals, return_inverse=True)

# 5) Loop & fit every frame
#    res columns = [A, x0, y0, s, B0, F_psf]
res = np.zeros((len(unique_times), 6))
for i in range(len(unique_times)):
    mask        = (inv_idx == i)
    frame_pids   = pix_vals[mask]
    frame_fluxes = flux_vals[mask]
    res[i]       = fit_frame_simple(frame_pids, frame_fluxes)

# 6) Compute per-frame pixel counts and residual RMS
pixel_counts = np.bincount(inv_idx, minlength=len(unique_times))
rms_resid    = np.full(len(unique_times), np.nan)

for i in range(len(unique_times)):
    if pixel_counts[i] < 10:
        continue
    # rebuild and background‐subtract
    z = np.zeros(n_pix)
    mask = (inv_idx == i)
    for pid, f in zip(pix_vals[mask], flux_vals[mask]):
        z[id2idx[pid]] = f
    z_sub = z - np.median(z)
    # model & residual
    A, x0, y0, s, B0, Fpsf = res[i]
    model   = gauss2d_circ((x_geom, y_geom), A, x0, y0, s)
    resid   = z_sub - model
    rms_resid[i] = np.sqrt(np.mean(resid**2))

# 7) Build “good” mask
good_pix = pixel_counts <= 3
good_rms = rms_resid < np.nanmedian(rms_resid) * 3
good     = good_pix & good_rms & (~np.isnan(res[:, -1]))

# 8) Plot the cleaned PSF photometry lightcurve
plt.figure(figsize=(8,4))
plt.plot(unique_times[good], res[good, -1], '-o', ms=3)
plt.xlabel('Time [s]')
plt.ylabel('Clean PSF‐fit Flux')
plt.title('Cleaned Circular‐Gaussian PSF Photometry Lightcurve')
plt.tight_layout()
plt.show()

# 9) (Optional) Smooth it
Fclean  = res[good, -1]
Fsmooth = savgol_filter(Fclean, window_length=11, polyorder=2)
plt.figure(figsize=(8,4))
plt.plot(unique_times[good], Fsmooth, linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Smoothed PSF Flux')
plt.title('Smoothed PSF Lightcurve')
plt.tight_layout()
plt.show()
#%%
plot_mean_flux_light_curve(tracks, track_number=0)
#%%
import numpy as np
import matplotlib.pyplot as plt

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
    plt.plot(binned_time_centers, binned_flux, linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Mean Flux")
    plt.title(f"Binned Mean-Flux Light Curve (Δt = {time_bin_size}s) for Track {track_number}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_binned_flux_light_curve(tracks, track_number=0, time_bin_size=50)
#%%

typed_nn_pix = get_typed_nn_pix(5)
print(typed_nn_pix)

new_high_pixel = read_single_run_utc_cut(161098)
pix = new_high_pixel[0]
tmp = new_high_pixel[1]
brightness = new_high_pixel[2]

tracks, possible_meteorites = track_sorter(pix, tmp, brightness, typed_nn_pix)

print(tracks)

for i, track in enumerate(tracks):
    pix = track[0]
    times = track[1]
    brightness = track[2]
    print(f"Track {i} — Unique Pixels: {len(set(pix))}")
    print(f"  First  → Pixel ID: {pix[0]}, Time: {times[0]}, Brightness: {brightness[0]}")
    print(f"  Last   → Pixel ID: {pix[-1]}, Time: {times[-1]}, Brightness: {brightness[-1]}")

plot_brightness_in_track(tracks)
plt.show()
#%%
plot_binned_flux_light_curve(tracks, track_number=0, time_bin_size=0.5)
#%%
def save_tracks_hits(run_number, output_file):
    """
    Processes a given run with track_sorter and writes every hit in each track
    as a row in a text file with columns: track_id, pixel_id, time[s], brightness.

    Parameters:
    - run_number: int or str, the run identifier
    - output_file: str, path to the output .txt file
    """
    # Build neighbor list
    typed_nn_pix = get_typed_nn_pix(5)

    # Read run data
    pix_arr, time_arr, bright_arr = read_single_run_utc_cut(run_number)

    # Detect tracks
    tracks, _ = track_sorter(pix_arr, time_arr, bright_arr, typed_nn_pix)

    # Write out hits
    with open(output_file, 'w') as f:
        # Header
        f.write("# track_id\tpixel_id\ttime[s]\tbrightness\n")
        # Loop tracks and hits
        for tid, tr in enumerate(tracks):
            pix_list = tr[0]
            time_list = tr[1]
            bright_list = tr[2]
            for p, t, b in zip(pix_list, time_list, bright_list):
                f.write(f"{tid}\t{int(p)}\t{t:.6f}\t{b:.6f}\n")

    print(f"Saved {sum(len(tr[0]) for tr in tracks)} hits from {len(tracks)} tracks to '{output_file}'")
#%%
save_tracks_hits(161098, "tracks_161098.txt")
#%%
typed_nn_pix = get_typed_nn_pix(5)
print(typed_nn_pix)

new_high_pixel = read_single_run_utc_cut(161098)
pix = new_high_pixel[0]
tmp = new_high_pixel[1]
brightness = new_high_pixel[2]

tracks, possible_meteorites = track_sorter(pix, tmp, brightness, typed_nn_pix)

print(tracks)

for i, track in enumerate(tracks):
    pix = track[0]
    times = track[1]
    brightness = track[2]
    print(f"Track {i} — Unique Pixels: {len(set(pix))}")
    print(f"  First  → Pixel ID: {pix[0]}, Time: {times[0]}, Brightness: {brightness[0]}")
    print(f"  Last   → Pixel ID: {pix[-1]}, Time: {times[-1]}, Brightness: {brightness[-1]}")

plot_brightness_in_track(tracks)
plt.show()
#%%
run_161098_tracks = tracks
#%%
plot_binned_flux_light_curve(run_161096_tracks, track_number=0, time_bin_size=50)
#%%
import numpy as np
import matplotlib.pyplot as plt

# List of (tracks, label) for the three runs
runs = [
    (run_161096_tracks, "run 161096"),
    (run_161097_tracks, "run 161097"),
    (run_161098_tracks, "run 161098"),
]

time_bin_size = 50  # in same units as your track times

fig, ax = plt.subplots(figsize=(10, 4))

for tracks, label in runs:
    # extract the 0th track
    pix_ids, times, brightnesses = tracks[0]
    times = np.array(times)
    brightnesses = np.array(brightnesses)
    
    # define bin edges
    t_min, t_max = times.min(), times.max()
    bins = np.arange(t_min, t_max + time_bin_size, time_bin_size)
    
    # digitize and compute binned means
    bin_idx = np.digitize(times, bins)
    centers, fluxes = [], []
    for i in range(1, len(bins)):
        mask = (bin_idx == i)
        if not mask.any():
            continue
        centers.append((bins[i-1] + bins[i]) / 2)
        fluxes.append(brightnesses[mask].mean())
    
    # plot on the same axes
    ax.plot(centers, fluxes, linestyle='-', label=label)

# finalize plot
ax.set_xlabel("Time")
ax.set_ylabel("Mean Brightness")
ax.set_title("Mean Light Curve [time bin-0.1s] for Track 0 (MGAB-V207) of Runs 161096, 161097, 161098")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
#%%
plot_binned_total_flux_light_curve(run_161096_tracks, track_number=0, time_bin_size=50)
#%%
import numpy as np
import matplotlib.pyplot as plt

# List of (tracks, label) for the three runs
runs = [
    (run_161096_tracks, "run 161096"),
    (run_161097_tracks, "run 161097"),
    (run_161098_tracks, "run 161098"),
]

time_bin_size = 50  # in same units as your track times

fig, ax = plt.subplots(figsize=(10, 4))

for tracks, label in runs:
    # extract the 0th track
    pix_ids, times, brightnesses = tracks[0]
    times = np.array(times)
    brightnesses = np.array(brightnesses)
    
    # define bin edges
    t_min, t_max = times.min(), times.max()
    bins = np.arange(t_min, t_max + time_bin_size, time_bin_size)
    
    # digitize and compute binned totals
    bin_idx = np.digitize(times, bins)
    centers, totals = [], []
    for i in range(1, len(bins)):
        mask = (bin_idx == i)
        if not mask.any():
            continue
        centers.append((bins[i-1] + bins[i]) / 2)
        totals.append(brightnesses[mask].sum())
    
    # plot on the same axes
    ax.plot(centers, totals, linestyle='-', label=label)

# finalize plot
ax.set_xlabel("Time")
ax.set_ylabel("Total Brightness")
ax.set_title("Total Light Curve [time bin-50s] for Track 0 (MGAB-V207) of Runs 161096, 161097, 161098")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# List of (tracks, label) for the three runs
runs = [
    (run_161096_tracks, "run 161096"),
    (run_161097_tracks, "run 161097"),
    (run_161098_tracks, "run 161098"),
]

time_bin_size = 50  # in same units as your track times

fig, ax = plt.subplots(figsize=(10, 4))

for tracks, label in runs:
    # extract the 0th track
    pix_ids, times, brightnesses = tracks[0]
    times = np.array(times)
    brightnesses = np.array(brightnesses)
    
    # define bin edges
    t_min, t_max = times.min(), times.max()
    bins = np.arange(t_min, t_max + time_bin_size, time_bin_size)
    
    # digitize and compute binned medians
    bin_idx = np.digitize(times, bins)
    centers, medians = [], []
    for i in range(1, len(bins)):
        mask = (bin_idx == i)
        if not mask.any():
            continue
        centers.append((bins[i-1] + bins[i]) / 2)
        medians.append(np.median(brightnesses[mask]))
    
    # plot on the same axes
    ax.plot(centers, medians, linestyle='-', label=label)

# finalize plot
ax.set_xlabel("Time")
ax.set_ylabel("Median Brightness")
ax.set_title("Median Light Curve [time bin-50s] for Track 0 (MGAB-V207) of Runs 161096, 161097, 161098")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
#%%
typed_nn_pix = get_typed_nn_pix(5)
print(typed_nn_pix)

new_high_pixel = read_single_run_utc_cut(161133)
pix = new_high_pixel[0]
tmp = new_high_pixel[1]
brightness = new_high_pixel[2]

tracks, possible_meteorites = track_sorter(pix, tmp, brightness, typed_nn_pix)

print(tracks)

for i, track in enumerate(tracks):
    pix = track[0]
    times = track[1]
    brightness = track[2]
    print(f"Track {i} — Unique Pixels: {len(set(pix))}")
    print(f"  First  → Pixel ID: {pix[0]}, Time: {times[0]}, Brightness: {brightness[0]}")
    print(f"  Last   → Pixel ID: {pix[-1]}, Time: {times[-1]}, Brightness: {brightness[-1]}")

plot_brightness_in_track(tracks)
plot_binned_mean_flux_light_curve(tracks, track_number=0, time_bin_size=50)
plt.show()
#%%
average_brightness(tracks, track_number=0)
#%%
def average_lightcurve_brightness(tracks, track_number=0):
    """
    For the given track, collapse multiple pixel brightnesses at each timestamp
    to a single mean, then return the average of those per-timestamp means.
    
    Parameters
    ----------
    tracks : list
        Each element is [pix_ids, times, brightnesses].
    track_number : int
        Index of the track to process.
    
    Returns
    -------
    float
        Mean of the per-timestamp mean brightnesses.
    """
    _, times, brightnesses = tracks[track_number]
    times = np.array(times)
    brightnesses = np.array(brightnesses)
    
    # Find unique timestamps and map each sample to its timestamp index
    uniq_times, inv = np.unique(times, return_inverse=True)
    
    # Sum brightnesses per timestamp, and count samples per timestamp
    sum_by_time   = np.bincount(inv, weights=brightnesses)
    count_by_time = np.bincount(inv)
    
    # Compute per-timestamp mean
    mean_by_time = sum_by_time / count_by_time
    
    # Finally, average those means
    return float(mean_by_time.mean())

mean_bright = average_lightcurve_brightness(tracks, track_number=0)
print(f"Mean brightness: {mean_bright:.3f}")
#%%
