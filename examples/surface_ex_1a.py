from __future__ import division

import argparse
import numpy as np
import tensorflow as tf
import progressbar
import imageio
import yaml
import matplotlib.pyplot as pp  # BJD added 18.11.2020
#import cv2 # BJD added 24.11.2020 - for make video
#import glob # BJD added 24.11.2020 - for make video
#import matplotlib.pyplot as plt
#import ffmpeg
import os # BJD added 24.11.2020 - for make video

import io # BJD added 18.11.2020
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from model_g import ModelG
from fluid_model_g import FluidModelG
from util import bl_noise
from numpy import * # BJD added 20.11.2020
from matplotlib import pyplot as plt # BJD added 20.11.2020

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np

dx = 2*R / args.height
x = (np.arange(args.width) - args.width // 2) * dx
y = (np.arange(args.height) - args.height // 2) * dx
x, y = np.meshgrid(x, y, indexing='ij')
