#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def cm(x, y):
    return (x/2.56, y/2.56)

# load the reference data
surf = np.genfromtxt("dat/ORCi_Drag_Surfaces.csv", delimiter=",", skip_header=1)
surf = surf.reshape((24, 43, 42))
# x is Fn := [0.,0.7], y is btr := [2.5,9], z is lvr := [3,9]
# we add zero Fn resistance (0.)
fn = np.hstack((0.0, np.linspace(0.125, 0.7, 24)))
btr = surf[0, 2:, 0]
lvr = surf[0, 1, 1:]
print("btr", btr)
print("lvr", lvr)

# build interpolation function for 3D data
data = np.zeros(((25, 41, 41)))
data[1:, :, :] = surf[:, 2:, 1:]

# fig, ax = plt.subplots(5,5,figsize=cm(20,20),subplot_kw={"projection": "3d"})
# X, Y = np.meshgrid(btr, lvr)
# for slab,ai in zip(data,ax.ravel()):
#     ai.plot_surface(X, Y, slab, linewidth=0, antialiased=False)
# plt.show()

# extrapolate if outside of range
# https://github.com/scipy/scipy/blob/v0.16.1/scipy/interpolate/interpolate.py#L1528
interp_Rr = RegularGridInterpolator(
    (fn, btr, lvr), data, method="linear", bounds_error=False, fill_value=None
)

# generate the data for training
btrs = np.linspace(0.25, 9.0, 32)
lvrs = np.linspace(3.0, 9.0, 32)

print(fn.flags)
data = np.empty((len(fn),len(btrs),len(lvrs)))
# Note:  To convert to drag in Newtons multiply the values by displacement and 9.81/1000.
for i in range(len(btrs)):
    for j in range(len(lvrs)):
        data[:,i,j] = interp_Rr((fn, btrs[i], lvrs[j]))
print(data.shape)
