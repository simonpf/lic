get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from matplotlib.colors import LogNorm
from mcrf.sensors import lcpr
from joint_flight.utils import despine_ax
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
try:
    plt.style.use("/home/simonpf/src/joint_flight/misc/matplotlib_style.rc")
except:
    pass
from netCDF4 import Dataset

from joint_flight.utils import iwc, rwc

################################################################################
# Retrieval results
################################################################################

retrieval_results = Dataset("/home/simonpf/src/crac/scripts/dardar/retrieval_results.nc", "r")
ice_dm = retrieval_results["Radar only"]["ice_dm"][:]
ice_n0 = retrieval_results["Radar only"]["ice_n0"][:]
rain_dm = retrieval_results["Radar only"]["rain_dm"][:]
rain_n0 = retrieval_results["Radar only"]["rain_n0"][:]
y = retrieval_results["Radar only"]["y_lcpr"][:]
yf = retrieval_results["Radar only"]["yf_lcpr"][:]
ice_md = iwc(ice_n0, ice_dm)
rain_md = rwc(rain_n0, rain_dm)

dardar_data = Dataset("dardar.nc", "r")
x_dardar = dardar_data["x"][:]
y_dardar = dardar_data["y"][:]
y_dardar = retrieval_results["Radar only"]["y_lcpr"][:]
lats_dardar = dardar_data["latitude"][:][:, 0]
lons_dardar = dardar_data["longitude"][:][:, 0]
z = np.linspace(0, 20e3, 41)
z_dardar = lcpr.range_bins
z_dardar = 0.5 * (z_dardar[1:] + z_dardar[:-1])
i_start = np.where(lats_dardar > -30)[0][0]
i_end = np.where(lats_dardar > -23)[0][0]
lats_dardar = lats_dardar[i_start : i_end]
lons_dardar = lons_dardar[i_start : i_end]
#z_dardar = z_dardar[i_start : i_end]


# Creating a custom RGB image
from matplotlib.colors import LogNorm
from matplotlib.cm import Reds, Blues
from matplotlib.gridspec import GridSpec

################################################################################
# y dardar
################################################################################

f = plt.figure(figsize = (10, 4))
gs = GridSpec(1, 2, width_ratios = [1.0, 0.03])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax.set_ylim([0, 20])

ax = plt.subplot(gs[0, 0])
img = ax.pcolormesh(lats_dardar, z_dardar / 1e3, y_dardar[i_start : i_end, :].T, norm = Normalize(-26, 20))
ax.set_xlabel(r"Latitude [$^\circ$]", fontsize = 12)
ax.set_ylabel(r"Altitude [km]", fontsize = 12)
ax.set_xlim([-30, -23])
despine_ax(ax, left = True, bottom = True)

ax = plt.subplot(gs[0, 1])
cb = plt.colorbar(img, cax = ax)
cb.ax.tick_params(labelsize=12)
cb.set_label("Radar reflectivity [dBZ]", fontsize = 12)

plt.tight_layout()
f.savefig("y_dardar.png", bbox_inches = "tight", dpi = 300)
plt.show()

def draw_retrieval(gs, ice = True):
    ax = plt.subplot(gs[1, 0])
    md = rain_md[i_start : i_end]
    cmap = Reds
    norm = LogNorm(1e-5, 1e-3)
    hue = cmap(norm(np.copy(md.T).ravel()))
    hue[md.T.ravel() < 1e-6, 3] = 0.0
    img = ax.pcolormesh(lats_dardar, z / 1e3, md.T, color = hue)
    img.set_array(None)

    md = ice_md[i_start : i_end]
    if ice:
        cmap = Blues
    else:
        cmap = Greys
    norm = LogNorm(1e-6, 1e-3)
    hue = cmap(norm(np.copy(md.T).ravel()))
    hue[md.T.ravel() < 1e-6, 3] = 0.0
    if ice:
        hue[:, 3] = 0.5
    img = ax.pcolormesh(lats_dardar, z / 1e3, md.T, color = hue, cmap = Blues)
    img.set_array(None)

    ax.set_xlabel(r"Latitude [$^\circ$]", fontsize = 12)
    ax.set_ylabel(r"Altitude [km]", fontsize = 12)
    ax.set_xlim([-30, -23])
    despine_ax(ax, left = True, bottom = True)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.set_ylim([0, 20])

    # Colorbar
    ax = plt.subplot(gs[1, 1])
    cb = plt.colorbar(img, cax = ax, cmap = Blues)
    cb.set_label(r"Mass content $[\unit{kg\ m^{-3}}]$", fontsize = 12)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ts = cb.get_ticks()
    xx = np.linspace(x0, x1, 101)
    yy = np.linspace(y0, y1, 101)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.logspace(-6, -3, 100)
    zz = np.broadcast_to(zz[np.newaxis, :], (100, 100))
    hue = Blues(norm(zz.T.ravel()))
    op = np.linspace(0, 1, 100)
    op = np.broadcast_to(op[:, np.newaxis], (100, 100))
    img = ax.pcolormesh(xx, yy, zz, color = hue, cmap = Reds)
    img.set_array(None)
    ax.set_yscale("linear")

    xx = np.linspace(x0, x1, 101)[58:]
    yy = np.linspace(y0, y1, 101)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.logspace(-6, -3, 100)
    zz = np.broadcast_to(zz[np.newaxis, :], (43, 100))
    hue = Reds(norm(zz.T.ravel()))
    #hue[:, 3] = op.T.ravel()
    img = ax.pcolormesh(xx, yy, zz.T, color = hue)

    cb.set_ticks(np.linspace(y0, y1, 4))
    cb.set_ticklabels([r"$10^{{-{}}}$".format(i) for i in range(4)])
    cb.ax.tick_params(labelsize=12)

f = plt.figure(figsize = (10, 6))
gs = GridSpec(2, 2, width_ratios = [1.0, 0.03])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax.set_ylim([0, 20])

ax = plt.subplot(gs[0, 0])
img = ax.pcolormesh(lats_dardar, z_dardar / 1e3, y_dardar[i_start : i_end, :].T, norm = Normalize(-26, 20))
ax.set_ylabel(r"Altitude [km]", fontsize = 12)
despine_ax(ax, left = True, bottom = False)

ax = plt.subplot(gs[0, 1])
cb = plt.colorbar(img, cax = ax)
cb.ax.tick_params(labelsize=12)
cb.set_label("Radar reflectivity [dBZ]", fontsize = 12)

draw_retrieval(gs)

plt.tight_layout()
f.savefig("retrieval_2.png", bbox_inches = "tight", dpi = 300)
plt.show()

################################################################################
# Simulated rain signal
################################################################################
from scipy.signal import convolve

simulation_results = Dataset("/home/simonpf/src/crac/scripts/dardar/forward_simulation_rain_full.nc", "r")
y_gmi = simulation_results["y_gmi"][:]
y_gmi_rain = y_gmi

k = np.ones(10) / 10.0

x = convolve(lats_dardar, k, "valid")

f = plt.figure(figsize = (10, 6))
gs = GridSpec(2, 2, width_ratios = [1.0, 0.03])

ax = plt.subplot(gs[0, 0])
gmi_indices = [0, 3, 4, 7]
handles = []
for ind in gmi_indices:
    y0 = y_gmi[0, ind]
    y = y_gmi[:, ind]
    dy = y - y0
    dy = convolve(dy, k, "valid")
    handles += ax.plot(x, dy)

ax.set_xlabel(r"Latitude [$^\circ$]", fontsize = 12)
ax.set_ylabel(r"$\Delta y$ [$\unit{K}$]", fontsize = 12)
ax.set_ylim([-60, 60])
ax.set_xlim([-30, -23])
despine_ax(ax, left = True, bottom = False)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax = plt.subplot(gs[0, 1])
ax.set_axis_off()
labels = [r"$10\ \unit{GHz}$", r"$37\ \unit{GHz}$", r"$89\ \unit{GHz}$", r"$183\ \unit{GHz}$"]
ax.legend(handles = handles, labels = labels, loc = "center left", fontsize = 12)

draw_retrieval(gs, ice = False)

plt.tight_layout()
f.savefig("signals_rain_2.pdf", bbox_inches = "tight")
plt.show()

################################################################################
# Ice signal
################################################################################

simulation_results = Dataset("/home/simonpf/src/crac/scripts/dardar/forward_simulation_ice_full.nc", "r")
y_gmi = simulation_results["y_gmi"][:]
y_ici = simulation_results["y_ici"][:]

f = plt.figure(figsize = (10, 6))
gs = GridSpec(2, 2, width_ratios = [1.0, 0.03])

ax = plt.subplot(gs[0, 0])
gmi_indices = [0, 3, 4, 7]
handles = []
for i, ind in enumerate(gmi_indices):
    y0 = y_gmi[0, ind]
    y = y_gmi[:, ind]
    dy = y - y0
    dy = convolve(dy, k, "valid")
    handles += ax.plot(x,  dy, c = "C{}".format(i))
    y0 = y_gmi_rain[0, ind]
    y = y_gmi_rain[:, ind]
    dy = y - y0
    dy = convolve(dy, k, "valid")
    ax.plot(x, dy, c = "C{}".format(i), ls = "--")

ax.set_xlabel(r"Latitude [$^\circ$]", fontsize = 12)
ax.set_ylabel(r"$\Delta y$ [$\unit{K}$]", fontsize = 12)
ax.set_ylim([-60, 60])
ax.set_xlim([-30, -23])
despine_ax(ax, left = True, bottom = False)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)


ax = plt.subplot(gs[0, 1])
ax.set_axis_off()
labels = [r"$10\ \unit{GHz}$", r"$37\ \unit{GHz}$", r"$89\ \unit{GHz}$", r"$183\ \unit{GHz}$"]
ax.legend(handles = handles, labels = labels, loc = "center left", fontsize = 12)

draw_retrieval(gs)
plt.tight_layout()
f.savefig("signals_ice_2.pdf", bbox_inches = "tight")
plt.show()

################################################################################
# ICI signal
################################################################################

f = plt.figure(figsize = (10, 6))
gs = GridSpec(2, 2, width_ratios = [1.0, 0.03])


ax = plt.subplot(gs[0, 0])
ici_indices = [3, 5, 8, 10]

handles = []

y0 = y_gmi[0, 7]
y = y_gmi[:, gmi_indices[-1]]
dy = y - y0
dy = convolve(dy, k, "valid")
handles += [ax.fill_between(x, 0.0, dy, color = "grey", alpha = 0.5, edgecolor = None)]

for ind in ici_indices:
    y0 = y_ici[0, ind]
    y = y_ici[:, ind]
    dy = y - y0
    dy = convolve(dy, k, "valid")
    handles += ax.plot(x, dy)

ax.set_ylabel(r"$\Delta y$ [$\unit{K}$]", fontsize = 12)
ax.set_ylim([-200, 0])
ax.set_xlim([-30, -23])
despine_ax(ax, left = True, bottom = False)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax = plt.subplot(gs[0, 1])
ax.set_axis_off()
labels = [r"$183\ \unit{GHz}$", r"$243\ \unit{GHz}$", r"$325\ \unit{GHz}$",
          r"$448\ \unit{GHz}$", r"$664\ \unit{GHz}$"]
ax.legend(handles = handles, labels = labels, loc = "center", fontsize = 12)
draw_retrieval(gs)

plt.tight_layout()
f.savefig("signals_ici_2.pdf", bbox_inches = "tight")
plt.show()
