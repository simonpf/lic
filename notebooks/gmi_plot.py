#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
try:
    plt.style.use("/home/simonpf/src/joint_flight/misc/matplotlib_style.rc")
except:
    pass


# In[2]:


get_ipython().run_line_magic('pwd', '')


# In[3]:


from datetime import datetime
import cloud_colocations
from cloud_colocations.colocations.products import set_cache
from cloud_colocations.colocations.formats import ModisCombined, DardarCloud, TWOBCLDCLASS, GPMGMI1C
set_cache("../data")

#t = datetime(2014, 3, 9, 1, 33)
#modis = ModisCombined.get_by_date(t)
#gpm = GPMGMI1C.get_by_date(t)
#dardar = DardarCloud.get_by_date(t)


# In[4]:


modis = ModisCombined("../data/MYD021KM.A2014068.0130.006.2014068155655.hdf",
                      "../data/MYD03.A2014068.0130.006.2014068151346.hdf")
gpm = GPMGMI1C("../data/1C-R.GPM.GMI.XCAL2016-C.20140309-S011445-E024711.000146.V05A.HDF5")
dardar = DardarCloud("../data/DARDAR-CLOUD_v2.1.1_2014068005444_41824.hdf")


lat_1 = -22
lat_0 = -28
lon_1 = -172
lon_0 = -178
center = np.array([0.5 * (lon_1 + lon_0), 0.5 * (lat_0 + lat_1)])

#
# Dardar
#

from geopy import distance
lats = dardar.get_latitudes()
lons = dardar.get_longitudes()
lons[lons > 0.0] -= 360
i_start, i_end = np.where(np.logical_and(lats > lat_0 - 4, lats <= lat_1 + 4))[0][[0, -1]]
plt.plot(lons, lats)
plt.plot(lons[i_start : i_end], lats[i_start : i_end])

lats_dardar = lats[i_start : i_end]
lons_dardar = lons[i_start : i_end]
x_dardar = np.zeros(i_end - i_start)
y_dardar = np.zeros(i_end - i_start)
for i in range(x_dardar.size):
    # c = [lat, lon]
    c = np.array([center[1], lons_dardar[i]])
    x_dardar[i] = np.sign(c[1] - center[0]) * distance.geodesic(c, center[::-1]).km
    c = np.array([lats_dardar[i], center[0]])
    y_dardar[i] = np.sign(c[0] - center[1]) * distance.geodesic(c, center[::-1]).km

plt.plot(x_dardar)

lats_dardar = lats[i_start : i_end]
lons_dardar = lons[i_start : i_end]
rr = dardar.get_radar_reflectivity()[i_start : i_end, :]
lb = dardar.get_lidar_backscatter()[i_start : i_end, :]
z = dardar.get_altitude()


from matplotlib.colors import LogNorm
plt.pcolormesh(x_dardar, z / 1e3, np.maximum(rr.T, 1e-8), norm = LogNorm())
get_ipython().run_line_magic('rm', 'dardar.nc')

from netCDF4 import Dataset
file = Dataset("dardar.nc", "w")
file.createDimension("along_track", lats_dardar.shape[0])
file.createDimension("z", z.size)
file.createDimension("dims", 3)
file.createVariable("z", "f4", dimensions = ("along_track", "z",))
file.createVariable("latitude", "f4", dimensions = ("along_track", "z"))
file.createVariable("longitude", "f4", dimensions = ("along_track", "z"))
file.createVariable("x", "f4", dimensions = ("along_track", "z"))
file.createVariable("y", "f4", dimensions = ("along_track", "z"))
file.createVariable("rr", "f4", dimensions = ("along_track", "z"))
file.createVariable("lb", "f4", dimensions = ("along_track", "z"))
file.createVariable("loc", "f4", dimensions = ("along_track", "z", "dims"))
file["z"][:] = np.broadcast_to(z[np.newaxis, :], rr.shape)
file["latitude"][:] = np.broadcast_to(lats_dardar[:, np.newaxis], rr.shape)
file["longitude"][:] = np.broadcast_to(lons_dardar[:, np.newaxis], rr.shape)
file["x"][:] = np.broadcast_to(x_dardar[:, np.newaxis], rr.shape)
file["y"][:] = np.broadcast_to(y_dardar[:, np.newaxis], rr.shape)
file["rr"][:] = rr
file["lb"][:] = lb
file["loc"][:, :, 0] = file["longitude"][:]
file["loc"][:, :, 1] = file["latitude"][:]
file["loc"][:, :, 2] = file["z"][:]
file.close()


#
# MODIS
#

lats = modis.geo_file.lats
lons = modis.geo_file.lons
i_start, i_end = np.where(np.logical_and(lats > lat_0, lats <= lat_1))[0][[0, -1]]
lats_modis = modis.geo_file.lats[i_start : i_end, :]
lons_modis = modis.geo_file.lons[i_start : i_end, :]
lons_modis[lons_modis >= 0] -= 360


# In[15]:


from skimage import exposure
modis_rgb = np.copy(np.transpose(modis.data[[0, 3, 2], i_start : i_end, :], [1, 2, 0]), order = "C")
for i in range(3):
    x_min = np.nanmin(modis_rgb[:, :, i])
    x_max = np.nanmax(modis_rgb[:, :, i])
    modis_rgb[:, :, i] = (modis_rgb[:, :, i] - x_min) / (x_max - x_min)
    #modis_rgb[:, :, i] = exposure.equalize_adapthist(modis_rgb[:, :, i], clip_limit=0.03)


# In[16]:

modis_c0 = modis.data[0, i_start : i_end, :]

from tqdm.auto import tqdm
x_modis = np.zeros((i_end - i_start, lats_modis.shape[1]))
y_modis = np.zeros((i_end - i_start, lats_modis.shape[1]))
for i in tqdm(range(x_modis.shape[0])):
    for j in range(x_modis.shape[1]):
        c = np.array([center[1], lons_modis[i, j]])
        x_modis[i, j] = np.sign(c[1] - center[0]) * distance.geodesic(c, center[::-1]).km
        c = np.array([lats_modis[i, j], center[0]])
        y_modis[i, j] = np.sign(c[0] - center[1]) * distance.geodesic(c, center[::-1]).km


# In[17]:


from netCDF4 import Dataset
file = Dataset("modis.nc", "w")
file.createDimension("along_track", lats_modis.shape[0])
file.createDimension("across_track", lats_modis.shape[1])
file.createDimension("channels", 3)
file.createVariable("latitude", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("longitude", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("x", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("y", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("true_color", "f4", dimensions = ("along_track", "across_track", "channels"))

file["latitude"][:] = lats_modis
file["longitude"][:] = lons_modis
file["x"][:] = x_modis
file["y"][:] = y_modis
file["true_color"][:] = modis_rgb
file.close()


# ## GPM data

# In[18]:


lats = gpm.lat_s1[:]
lons = gpm.lon_s1[:]
lons[lons >= 0.0] -= 360
i_start, i_end = np.where(np.logical_and(lons > lon_0, lons <= lon_1))[0][[0, -1]]
#i_end = i_start + 500

tbs_gpm = np.zeros((i_end - i_start, lats.shape[1], 13))
lats_gpm = gpm.lat_s1[i_start : i_end, :]
lons_gpm = gpm.lon_s1[i_start : i_end, :]
lons_gpm[lons_gpm >= 0.0] -= 360
tbs_gpm[:, :, :9] = gpm.y_s1[i_start : i_end, :, :]
tbs_gpm[:, :, 9:] = gpm.y_s2[i_start : i_end, :, :]


# In[19]:


from tqdm.auto import tqdm
x_gpm = np.zeros((i_end - i_start, lats_gpm.shape[1]))
y_gpm = np.zeros((i_end - i_start, lats_gpm.shape[1]))
for i in tqdm(range(x_gpm.shape[0])):
    for j in range(x_gpm.shape[1]):
        c = np.array([center[1], lons_gpm[i, j]])
        x_gpm[i, j] = np.sign(c[1] - center[0]) * distance.geodesic(c, center[::-1]).km
        c = np.array([lats_gpm[i, j], center[0]])
        y_gpm[i, j] = np.sign(c[0] - center[1]) * distance.geodesic(c, center[::-1]).km


# In[20]:


get_ipython().run_line_magic('rm', 'gpm.nc')


# In[21]:


from netCDF4 import Dataset
file = Dataset("gpm.nc", "w")
file.createDimension("along_track", lats_gpm.shape[0])
file.createDimension("across_track", lats_gpm.shape[1])
file.createDimension("channels", 13)
file.createVariable("latitude", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("longitude", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("x", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("y", "f4", dimensions = ("along_track", "across_track"))
file.createVariable("tbs", "f4", dimensions = ("along_track", "across_track", "channels"))

file["latitude"][:] = lats_gpm
file["longitude"][:] = lons_gpm
file["x"][:] = x_gpm
file["y"][:] = y_gpm
file["tbs"][:] = tbs_gpm
file.close()


# ## Putting it together

# In[100]:


rr.shape


# In[105]:


y_modis.shape

# In[103]:

import pyvista as pv
from matplotlib.colors import Normalize
from matplotlib.cm import magma
from cloud_colocations.plots import grid_to_edges

def make_dardar(i_start = None, i_end = None):
    zif i_start is None:
        i_start = np.where(y_dardar > y_modis.min())[0][0]
    if i_end is None:
        i_end = np.where(y_dardar > y_modis.max())[0][-1]
    rrr = rr[i_start : i_end, :]
    xx = grid_to_edges(np.broadcast_to(x_dardar[i_start: i_end, np.newaxis], rrr.shape))
    yy = grid_to_edges(np.broadcast_to(y_dardar[i_start: i_end, np.newaxis], rrr.shape))
    zz = grid_to_edges(np.broadcast_to(z[np.newaxis, :], rrr.shape)) / 1e2
    vertices = np.zeros((xx.size, 3))
    vertices[:, 0] = xx.ravel()
    vertices[:, 1] = yy.ravel()
    vertices[:, 2] = zz.ravel()
    dardar_curtain = pv.StructuredGrid(xx, yy, zz)
    dbz =  np.minimum(10.0 * np.log10(np.maximum(rrr[:, :], 1e-3)).T, 20)
    #dbz =  np.copy(yy[1::, 1:].T, order=  "F")
    dardar_curtain.cell_arrays["radar_reflectivity"] = dbz.ravel()
    dardar_curtain.save("dardar.vts")
    dardar_curtain.set_active_scalar("radar_reflectivity")
    return dardar_curtain


def make_modis(j_start = 0, j_end = -1):
    xx = grid_to_edges(x_modis[:, j_start : j_end])
    yy = grid_to_edges(y_modis[:, j_start : j_end])
    zz = grid_to_edges(np.zeros(x_modis[:, j_start : j_end].shape))
    vertices = np.zeros((xx.size, 3))
    vertices[:, 0] = xx.ravel()
    vertices[:, 1] = yy.ravel()
    vertices[:, 2] = zz.ravel()
    modis_surface = pv.StructuredGrid(xx, yy, zz)

    m = xx.shape[0]
    n = xx.shape[1]
    tc_x = np.linspace(1, 0, m).reshape(-1, 1)
    tc_y = np.linspace(0, 1, n).reshape(1, -1)
    tc_x, tc_y = np.meshgrid(tc_x, tc_y)
    tcs = np.zeros((m * n, 2))
    tcs[:, 1] = tc_x.ravel()
    tcs[:, 0] = tc_y.ravel()

    origin = [x_modis[0, 0], y_modis[0, 0], zz[0, 0]]
    u = [x_modis[0, -1], y_modis[0, -1], zz[0, -1]]
    v = [x_modis[-1, 0], y_modis[-1, 0], zz[-1, 0]]
    modis_surface.texture_map_to_plane(origin, u, v, inplace = True)
    #modis_surface.cell_arrays["Texture Coordinates"] = tcs
    modis_texture = pv.numpy_to_texture(np.array(256 * modis_rgb[:, j_start : j_end, :], order = "F", dtype = np.uint8))
    modis_surface.point_arrays["Texture Coordinates"][:, 0] = tcs[:, 0]
    modis_surface.point_arrays["Texture Coordinates"][:, 1] = tcs[:, 1]

    return modis_surface, modis_texture


def make_gpm(i_start = 50,
             i_end = 200,
             j_start = 50,
             j_end = 170,
             channels = [1, 6, 8, 12],
             cf = 0.5):

    n_channels = len(channels)

    xx = x_gpm[i_start : i_end, j_start : j_end]
    yy = y_gpm[i_start : i_end, j_start : j_end]
    zz = np.ones(x_gpm[i_start : i_end, j_start : j_end].shape)
    tbs = tbs_gpm[i_start : i_end, j_start : j_end, :]

    texture = []

    from matplotlib.cm import magma
    cm = magma
    norm = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

    surfaces = []
    textures = []

    for i, c in enumerate(channels):

        m = i_end - i_start
        n = j_end - j_start

        i_start_c = 0
        i_end_c = -1 - (n_channels - i - 1) * int(cf * m) // n_channels
        j_start_c = i * (n // n_channels)
        j_end_c = min(j_start_c + (n // n_channels), n)
        print(j_start_c, j_end_c, n)

        # Surface
        xx_c = grid_to_edges(xx[i_start_c : i_end_c, j_start_c : j_end_c])
        yy_c = grid_to_edges(yy[i_start_c : i_end_c, j_start_c : j_end_c])
        zz_c = grid_to_edges(zz[i_start_c : i_end_c, j_start_c : j_end_c])
        vertices = np.zeros((xx_c.size, 3))
        vertices[:, 0] = xx_c.ravel()
        vertices[:, 1] = yy_c.ravel()
        vertices[:, 2] = zz_c.ravel()
        surface = pv.StructuredGrid(xx_c, yy_c, zz_c)

        m = xx_c.shape[0]
        n = xx_c.shape[1]
        tc_x = np.linspace(1, 0, m).reshape(-1, 1)
        tc_y = np.linspace(0, 1, n).reshape(1, -1)
        tc_x, tc_y = np.meshgrid(tc_x, tc_y)
        tcs = np.zeros((m * n, 2))
        tcs[:, 1] = tc_x.ravel()
        tcs[:, 0] = tc_y.ravel()

        origin = [xx_c[0, 0], yy_c[0, 0], zz_c[0, 0]]
        u = [xx_c[0, -1], yy_c[0, -1], zz_c[0, -1]]
        v = [xx_c[-1, 0], yy_c[-1, 0], zz_c[-1, 0]]
        surface.texture_map_to_plane(origin, u, v, inplace = True)
        surface.point_arrays["Texture Coordinates"][:, 0] = tcs[:, 0]
        surface.point_arrays["Texture Coordinates"][:, 1] = tcs[:, 1]
        surfaces += [surface]

        # Texture

        y = tbs[i_start_c : i_end_c, j_start_c : j_end_c, c]
        hue = norm(np.copy(y.ravel()))
        print(xx_c.shape, y.shape)
        colors = (cm(hue)[:, 0:3] * 255.0).astype(np.uint8)
        image = colors.reshape((xx_c.shape[0] - 1, xx_c.shape[1] - 1, 3))
        textures += [pv.numpy_to_texture(image)]

    return surfaces, textures


c_dardar = [np.mean(x_dardar), np.mean(y_dardar)]
d_modis = np.sqrt((x_modis - c_dardar[0]) ** 2 + (y_modis - c_dardar[1]) ** 2)
ci_modis = np.argmin(d_modis.ravel())
ic_modis = ci_modis // x_modis.shape[1]
jc_modis = ci_modis % x_modis.shape[1]


dardar_curtain = make_dardar()
modis_surface, modis_texture = make_modis(j_start = ic_modis - 400, j_end = jc_modis + 400)
gpm_surfaces, gpm_textures = make_gpm(i_start = 80, i_end = 160, j_start = 70, j_end = 150)

plotter = pv.Plotter()
cp = plotter.camera_position
#plotter.camera_posistion = [(c[0] / 2.0, c[1] / 2.0, c[2] / 2.0) for c in cp]
#cp_new = plotter.camera_posistion
plotter.background_color = "white"
plotter.add_mesh(modis_surface, texture = modis_texture)
plotter.add_mesh(dardar_curtain, lighting = False, opacity = "sigmoid")
plotter.add_mesh(gpm_surfaces[0], lighting = False, texture = gpm_textures[0], opacity = "sigmoid")
plotter.add_mesh(gpm_surfaces[1], lighting = False, texture = gpm_textures[1])
plotter.add_mesh(gpm_surfaces[2], lighting = False, texture = gpm_textures[2])
plotter.add_mesh(gpm_surfaces[3], lighting = False, texture = gpm_textures[3])
plotter.add_axes()
plotter.add_bounding_box(color = "black")
plotter.show_bounds(color = "black")
plotter.show()


f, axs = plt.subplots(1, 2)
image = np.array(modis_rgb, order = "C")
m, n, _ = image.shape
c = image.transpose((0, 1, 2)).reshape(m * n, 3)

xx = grid_to_edges(x_modis)
yy = grid_to_edges(y_modis)
img = axs[0].pcolormesh(xx, yy, modis_rgb[:, :, 1], color = c)
img.set_array(None)
axs[0].scatter(x_dardar, y_dardar, marker = "x", c = "k")
axs[1].pcolormesh(xx, yy, c.reshape(modis_rgb.shape)[:, :, 0])
axs[1].scatter(x_dardar, y_dardar, marker = "x", c = "k")
#axs[1].pcolormesh(x_dardar, z, 10 * np.log10(np.maximum(10 ** -2.5, rr.T)))



