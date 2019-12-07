get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import Normalize
from matplotlib.cm import magma
from cloud_colocations.plots import grid_to_edges
try:
    plt.style.use("/home/simonpf/src/joint_flight/misc/matplotlib_style.rc")
except:
    pass

from netCDF4 import Dataset

#
# Modis
#

modis_data = Dataset("modis.nc", "r")
x_modis = modis_data["x"][:][::10, ::10]
y_modis = modis_data["y"][:][::10, ::10]
xx = grid_to_edges(x_modis[:, :])
yy = grid_to_edges(y_modis[:, :])
zz = grid_to_edges(np.zeros(x_modis[:, :].shape))
vertices = np.zeros((xx.size, 3))
vertices[:, 0] = xx.ravel()
vertices[:, 1] = yy.ravel()
vertices[:, 2] = zz.ravel()
modis_surface = pv.StructuredGrid(xx, yy, zz)
modis_texture = pv.read_texture("gmi_texture_[5, 6].png")
modis_surface.texture_map_to_plane(inplace = True)

m = xx.shape[0]
n = xx.shape[1]
tc_x = np.linspace(0, 1, m).reshape(-1, 1)
tc_y = np.linspace(0, 1, n).reshape(1, -1)
tc_x, tc_y = np.meshgrid(tc_x, tc_y, indexing = "xy")
tcs = np.zeros((m * n, 2))
tcs[:, 0] = tc_x.T.ravel(order = "F")
tcs[:, 1] = tc_y.T.ravel(order = "F")
modis_surface.point_arrays["Texture Coordinates"][:, 0] = tcs[:, 0]
modis_surface.point_arrays["Texture Coordinates"][:, 1] = tcs[:, 1]

#
# Dardar
#

dardar_data = Dataset("dardar.nc", "r")
x_dardar = dardar_data["x"][:]
y_dardar = dardar_data["y"][:]
z_dardar = dardar_data["z"][:]

i_start = np.where(y_dardar > -600)[0][0]
i_end = np.where(y_dardar > 600)[0][0]
dbz = np.minimum(10 * np.log10(np.maximum(dardar_data["rr"], 10 ** -2.6)), 20)[i_start : i_end]

xx = grid_to_edges(x_dardar[i_start : i_end, :])
yy = grid_to_edges(y_dardar[i_start : i_end, :])
zz = grid_to_edges(z_dardar[i_start : i_end, :]) / 5e1

for i in range(dbz.shape[0]):
    z = zz[i, :]
    j = np.where(z < 15)[0][0]
    dbz[i, j:] = dbz[i, j]

vertices = np.zeros((xx.size, 3))
vertices[:, 0] = xx.ravel()
vertices[:, 1] = yy.ravel()
vertices[:, 2] = zz.ravel()
dardar_curtain = pv.StructuredGrid(xx, yy, zz)
#dbz =  np.copy(yy[1::, 1:].T, order=  "F")
dardar_curtain.cell_arrays["radar_reflectivity"] = dbz.T.ravel()
dardar_curtain.save("dardar.vts")
dardar_curtain.set_active_scalar("radar_reflectivity")

#m = xx.shape[0]
#n = xx.shape[1]
#tc_x = np.linspace(1, 0, m).reshape(-1, 1)
#tc_y = np.linspace(0, 1, n).reshape(1, -1)
#tc_x, tc_y = np.meshgrid(tc_x, tc_y)
#tcs = np.zeros((m * n, 2))
#tcs[:, 1] = tc_x.ravel()
#tcs[:, 0] = tc_y.ravel()
#
#origin = [x_modis[0, 0], y_modis[0, 0], zz[0, 0]]
#u = [x_modis[0, -1], y_modis[0, -1], zz[0, -1]]
#v = [x_modis[-1, 0], y_modis[-1, 0], zz[-1, 0]]
##modis_surface.cell_arrays["Texture Coordinates"] = tcs
#modis_texture = pv.numpy_to_texture(np.array(256 * modis_rgb[:, j_start : j_end, :], order = "F", dtype = np.uint8))
#modis_surface.point_arrays["Texture Coordinates"][:, 0] = tcs[:, 0]
#modis_surface.point_arrays["Texture Coordinates"][:, 1] = tcs[:, 1]
#
#return modis_surface, modis_texture

#bounds = [700, 1500, -800, 800, -100, 100]
#modis_clipped = modis_clipped.clip_box(bounds)
#bounds = [-1500, 1500, -800, -600, -100, 100]
#modis_clipped = modis_clipped.clip_box(bounds)
#bounds = [-1500, 1500, 600, 1000, -100, 100]
#modis_clipped = modis_clipped.clip_box(bounds)
bounds = [-2000, -900, -800, 800, -100, 100]
modis_clipped = modis_surface.clip_box(bounds)

plotter = pv.BackgroundPlotter()
#dardar_clipped = dardar_curtain.clip_surface(modis_surface)
plotter.add_mesh(modis_clipped, texture = modis_texture, lighting = False)
plotter.add_mesh(dardar_curtain, lighting = False, opacity = "sigmoid", cmap = "magma", show_scalar_bar = False)
#plotter.add_bounding_box(color = "black")
#plotter.show_grid(color = "black")
plotter.background_color = "white"
plotter.show(screenshot = "cloudsat_gmi.png")

