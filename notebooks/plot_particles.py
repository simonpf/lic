get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from joint_flight.utils import despine_ax
try:
    plt.style.use("/home/simonpf/src/joint_flight/misc/matplotlib_style.rc")
except:
    pass

img_15  = "/home/simonpf/src/joint_flight/data/particle_images_100/cip_image_20161014_105050_087.png"
img_100 = "/home/simonpf/src/joint_flight/data/particle_images_15/cip_image_20161014_110134_070.png"


f, axs = plt.subplots(2, 1, figsize = (10, 4))

img = sp.misc.imread(img_15)
x = 0.015 * np.arange(img.shape[1])
y = 0.015 * np.arange(img.shape[0])

ax = axs[0]
ax.pcolormesh(x, y, img, cmap = "bone")
ax.set_ylabel(r"$y\ [\unit{mm}]$", fontsize = 14)
ax.set_aspect(1)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)


img = sp.misc.imread(img_100)
x = 0.1 * np.arange(img.shape[1])
y = 0.1 * np.arange(img.shape[0])
ax = axs[1]
ax.pcolormesh(x, y, img, cmap = "bone")
ax.set_xlabel(r"$x\ [\unit{mm}]$", fontsize = 14)
ax.set_ylabel(r"$y\ [\unit{mm}]$", fontsize = 14, labelpad=15)
ax.set_aspect(1)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)

plt.tight_layout()
