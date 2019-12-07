get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
try:
    plt.style.use("/home/simonpf/src/joint_flight/misc/matplotlib_style.rc")
except:
    pass

from netCDF4 import Dataset

modis_data = Dataset("modis.nc", "r")
gmi_data = Dataset("gpm.nc", "r")

image = modis_data["true_color"][:, :, :]
x = np.arange(image.shape[0])
y = np.arange(image.shape[1])

fig = plt.figure(frameon = False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(image)
fig.savefig("modis_image.jpg", dpi = 200)

import matplotlib.image as mpimg
from scipy.interpolate import griddata

lats_modis = modis_data["latitude"][:]
lons_modis = modis_data["longitude"][:]
lats_gmi = gmi_data["latitude"][:]
lons_gmi = gmi_data["longitude"][:]
tbs = gmi_data["tbs"][:]

image = mpimg.imread('modis_image.png')
x = np.linspace(0.0, image.shape[0] - 1.0, lons_modis.shape[0])
y = np.linspace(0.0, image.shape[1] - 1.0, lons_modis.shape[1])
xx, yy = np.meshgrid(x, y, indexing = 'ij')

points = np.zeros((lats_modis.size, 2))
points[:, 0] = lats_modis.ravel()
points[:, 1] = lons_modis.ravel()

f_x = griddata(points, xx.ravel(order = "C"), (lats_gmi.ravel(order = "C"), lons_gmi.ravel(order = "C")))
f_y = griddata(points, yy.ravel(order = "C"), (lats_gmi.ravel(order = "C"), lons_gmi.ravel(order = "C")))
x_gmi = f_x.reshape(lats_gmi.shape, order = "C")
y_gmi = f_y.reshape(lats_gmi.shape, order = "C")


def plot_single_channel(filename,
                        ci = 1,
                        j_start = 0,
                        j_end = -1):
    plt.ioff()
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    xi = np.arange(image.shape[0])
    yi = np.arange(image.shape[1])

    colors = np.array(image[:-1, :-1, :].transpose((1, 0, 2)), order = "C").reshape((image.shape[0] - 1) * (image.shape[1] - 1),  3, order = "C")
    img = ax.pcolormesh(xi, yi, image[:, :, 0].T, color = colors)
    #img = ax.pcolormesh(xi, yi, image[:, :, 0].T)
    img.set_array(None)

    tbs_masked = np.copy(tbs)
    inds = np.logical_or(np.isnan(x_gmi), np.isnan(y_gmi))
    tbs_masked[inds] = np.nan

    xx = x_gmi[:, j_start : j_end]
    yy = y_gmi[:, j_start : j_end]

    if type(ci) == list:
        y = np.sum(tbs[:, j_start:j_end, ci], axis = -1)
    else:
        y = tbs[:, j_start:j_end, ci]

    k = np.ones((5, 5)) / 25.0
    y = convolve(y, k, "valid")
    xx = xx[2:-2, 2:-2]
    yy = yy[2:-2, 2:-2]


    maxxx = np.max(y[y > 100])
    minnn = np.min(y[y > 100])
    y = np.maximum(minnn * 1.001, y)
    levels = np.linspace(minnn, maxxx, 5)


    print(xx)
    print(yy)
    print(y)
    print(levels)

    ax.contour(xx, yy, y, levels = 5, linewidths = 1.5)
    ax.plot(xx[:, -1], yy[:, -1], c = "k", ls = "--", lw = 1.2)
    ax.plot(xx[:, 0], yy[:, 0], c = "k", ls = "--", lw = 1.2)
    fig.savefig(filename, dpi = 300)
    plt.close()
    plt.ion()

for inds in [[0, 1], [2, 3], 4, [5, 6], [7, 8], [9, 10], 11, 12]:
    plot_single_channel("gmi_texture_{}.png".format(inds), ci = inds)
    plot_single_channel("gmi_texture_{}_thin.png".format(inds), ci = inds, j_start = 25, j_end = 199)

fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

xi = np.arange(image.shape[0])
yi = np.arange(image.shape[1])

colors = np.array(image[:-1, :-1, :].transpose((1, 0, 2)), order = "C").reshape((image.shape[0] - 1) * (image.shape[1] - 1),  3, order = "C")
img = ax.pcolormesh(xi, yi, image[:, :, 0].T, color = colors)
#img = ax.pcolormesh(xi, yi, image[:, :, 0].T)
img.set_array(None)

tbs_masked = np.copy(tbs)
inds = np.logical_or(np.isnan(x_gmi), np.isnan(y_gmi))
tbs_masked[inds] = np.nan

j_start = 0
j_end = -1
xx = x_gmi[:, j_start : j_end]
yy = y_gmi[:, j_start : j_end]
y = tbs_masked[:, j_start : j_end]
ax.contour(yy, xx, y[:, :, 0], levels = 5, linewidths = 1)
ax.plot(yy[:, -1], xx[:, -1], c = "k", ls = "--", lw = 1)
ax.plot(yy[:, 0], xx[:, 0], c = "k", ls = "--", lw = 1)

################################################################################
# MODIS + GMI co-locations
################################################################################

from skimage import exposure

f, axs = plt.subplots(2, 2, figsize = (16, 10))
channels = [[0, 1], [5, 6], [7, 8], [12]]

modis_start = 100
modis_end = -200
img = modis_data["true_color"][:][:, modis_start : modis_end, :]
m, n,_ = img.shape

img_corrected = exposure.adjust_log(img, 1)

colors = img_corrected[:-1, :-1 :].transpose((0, 1, 2)).reshape((m - 1) * (n - 1), 3, order = "C")

from joint_flight.utils import despine_ax

titles = [r"(a) $10\ \unit{GHz}$",
          r"(b) $37\ \unit{GHz}$",
          r"(c) $89\ \unit{GHz}$",
          r"(c) $183\ \unit{GHz}$"]
for i, c in enumerate(channels):
    ax = axs[i // 2, i % 2]

    im = ax.pcolormesh(lons_modis[:, modis_start : modis_end],
                       lats_modis[:, modis_start : modis_end],
                       img[:-1, :-1, 0], color = colors)
    im.set_array(None)


    gpm_start = 50
    gpm_end = -50
    y = np.sum(tbs[gpm_start : gpm_end, 25:199, c], axis = -1)

    k = np.ones((5, 5)) / 25.0
    y = convolve(y, k, "valid")
    xx = lons_gmi[gpm_start : gpm_end, 25:199]
    xx = convolve(xx, k, "valid")
    yy = lats_gmi[gpm_start : gpm_end, 25:199]
    yy = convolve(yy, k, "valid")

    maxxx = np.max(y[y > 150])
    minnn = np.min(y[y > 150])
    y = np.maximum(minnn * 1.001, y)
    levels = np.linspace(minnn, maxxx, 5)
    ax.contour(xx, yy, y, levels = levels, linewidths = 2)

    ax.set_xlim([-180, -170])
    ax.set_ylim([-32, -20])
    ax.set_title(titles[i], loc = "left")
    ax.plot(lons_gmi[gpm_start : gpm_end, 25], lats_gmi[gpm_start : gpm_end, 25], c = "grey", ls = "--")
    ax.plot(lons_gmi[gpm_start : gpm_end, 199], lats_gmi[gpm_start : gpm_end, 199], c = "grey", ls = "--")
    despine_ax(ax, left = i % 2 == 0, bottom = i // 2 == 1)
    if i // 2 == 1:
        ax.set_xlabel(r"Longitude\ [$^\circ$]")
    if i % 2 == 0:
        ax.set_ylabel(r"Latitude\ [$^\circ$]")
plt.tight_layout()
f.savefig("modis_gmi.png", dpi = 300, bbox_inches = "tight")

################################################################################
# MODIS + CloudSat
################################################################################

dardar_data = Dataset("dardar.nc", "r")
x_dardar = dardar_data["x"][:]
y_dardar = dardar_data["y"][:]

lats_dardar = dardar_data["latitude"][:]
lons_dardar = dardar_data["longitude"][:]
z_dardar = dardar_data["z"][:]
i_start = np.where(lats_dardar > -30)[0][0]
i_end = np.where(lats_dardar > -23)[0][0]
lats_dardar = lats_dardar[i_start : i_end]
lons_dardar = lons_dardar[i_start : i_end]
z_dardar = z_dardar[i_start : i_end]


dbz = np.minimum(10 * np.log10(np.maximum(dardar_data["rr"], 10 ** -2.6)), 20)[i_start : i_end]

from matplotlib.gridspec import GridSpec
from joint_flight.utils import despine_ax
f = plt.figure(figsize = (16, 6))
gs = GridSpec(1, 3, width_ratios = [1, 1, 0.03])

channels = [1, 6, 8, 12]
modis_start = 100
modis_end = -200
img = modis_data["true_color"][:][:, modis_start : modis_end, :]
m, n,_ = img.shape
colors = img[:-1, :-1 :].transpose((0, 1, 2)).reshape((m - 1) * (n - 1), 3, order = "C")

titles = [r"(a) True-color composite",
          r"(b) Radar observations"]

ax = plt.subplot(gs[0])
im = ax.pcolormesh(lons_modis[:, modis_start : modis_end],
                    lats_modis[:, modis_start : modis_end],
                    img[:-1, :-1, 0], color = colors)
im.set_array(None)
ax.plot(lons_dardar, lats_dardar, c = "C0", ls = "--")
ax.scatter(lons_dardar[[0, -1]], lats_dardar[[0, -1]], marker = "x", zorder = 20, c = "C0")
#ax.annotate("a", [lons_dardar[0], lats_dardar[0]], color = "C0")
#ax.annotate("b", [lons_dardar[-1], lats_dardar[-1]], color = "C0")
ax.set_xlim([-180, -170])
ax.set_ylim([-32, -20])
ax.set_title(titles[0], loc = "left")
despine_ax(ax, left = True, bottom = True)
ax.set_xlabel(r"Longitude\ [$^\circ$]")
ax.set_ylabel(r"Latitude\ [$^\circ$]")

ax = plt.subplot(gs[1])
im = ax.pcolormesh(lats_dardar[:, :],
                   z_dardar[:, :] / 1e3,
                   dbz)
ax.set_ylim([0, 15])
ax.set_xlim([-30, -23])
ax.set_title(titles[1], loc = "left")
despine_ax(ax, left = True, bottom = True)
ax.set_xlabel(r"Latitude [$^\circ$]")
ax.set_ylabel(r"Altitude [km]")

ax = plt.subplot(gs[2])
plt.colorbar(im, cax = ax, label = "Radar reflectivity [dBZ]")
plt.tight_layout()
f.savefig("modis_cloudsat.png", dpi = 300, bbox_inches = "tight")
