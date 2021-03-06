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
from mpl_toolkits.mplot3d import Axes3D # BJD added 20.11.2020
from matplotlib import cm # BJD added 20.11.2020

from scipy import signal # BJD added 14.1.2021


RESOLUTIONS = {
    "2160p": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
    "240p": (426, 240),
    "160p": (284, 160),
    "80p": (142, 80),
    "40p": (71, 40),
}


def make_video_frame(rgb, indexing='ij'):
    if indexing == 'ij':
        rgb = [tf.transpose(channel) for channel in rgb]
    frame = tf.stack(rgb, axis=-1)
    frame = tf.clip_by_value(frame, 0.0, 1.0)
    return tf.cast(frame * 255, 'uint8').numpy()

#......................................................
def nucleation_and_motion_in_G_gradient_fluid_2D(writer, args, R=8): # was R=60, 30 and 15 -BJD 14.1.2021
    c1 = 0 # BJD added this on 20.11.2020
    dx = 2*R / args.height
    x = (np.arange(args.width) - args.width // 2) * dx
    y = (np.arange(args.height) - args.height // 2) * dx
    x, y = np.meshgrid(x, y, indexing='ij')

    #def trapezoid_signal(x, width=2., slope=1., amp=1., offs=0):
    def trapezoid_signal(x, width=2., slope=1., amp=10., offs=1):
        #a = 10 * slope*width*signal.sawtooth(2 * np.pi * 1/10* x/width, width=0.5)/4.
        #a = slope * width * signal.sawtooth(2 * np.pi * 1/10 * x/width, width=0.5)/4.
        #a = slope * width * signal.sawtooth(2 * np.pi * 1/10 * x/width, width=0.5)/4.
        a = slope * width * signal.sawtooth(2 * np.pi * 1/10 * x/width - 1, width=0.5)/4.
        a[a>amp/2.] = amp/2.
        a[a<-amp/2.] = -amp/2.
        return a + amp/2. + offs

    def f(x):
        #return -3*x/10 #+ 20
        neg = x < 0
        neg2 = np.logical_and(x >= 0, x < 4.7) #c[a & b] x >= 0 and x < 20  c[a & b]  np.logical_and(x > -2, x < 2)
        neg3 = x >= 4.7
        return neg * (-x/5 + 1) + neg2 * (50 * signal.square(2 * np.pi /10 * x)) + neg3 * 1

    def f2(x):
        #return -3*x/10 #+ 20
        neg4 = x < 0
        #neg2 = np.logical_and(x >= 0, x < 4.7) #c[a & b] x >= 0 and x < 20  c[a & b]  np.logical_and(x > -2, x < 2)
        #neg3 = x >= 4.7
        neg5 = x >= 0
        #return neg * (-x/5 + 1) + neg2 * (50 * signal.square(2 * np.pi /10 * x)) + neg3 * 1
        return neg4 * (-x/5 + 1) + neg5 * 1

    def f3(x):
        #return -3*x/10 #+ 20
        #neg = x < 0
        #neg6 = np.logical_and(y >= -10, y <= 10)
        #neg7 = np.logical_and(y < -10, y > 10)
        neg6 = np.logical_and(y >= -5, y <= 5)
        neg7 = np.logical_or(y < -5, y > 5)
        #neg2 = np.logical_and(x >= 0, x < 4.7) #c[a & b] x >= 0 and x < 20  c[a & b]  np.logical_and(x > -2, x < 2)
        #neg3 = x >= 4.7
        #neg3 = x >= 0
        #return neg * (-x/5 + 1) + neg2 * (50 * signal.square(2 * np.pi /10 * x)) + neg3 * 1
        return neg6 * f2(x) + neg7 * f(x)
    #...........................................

    def source_G(t):
        #center = np.exp(-0.5*(t-5)**2) * 10
        center = np.exp(-0.5*(t-5)**2) * 10
        gradient = (1+np.tanh(t-30)) * 0.0003
        #potential = 0.015 * (np.tanh(t-25) + 1)
        #x = np.linspace(-40, 40, 500)
        #triangle = 10 * signal.sawtooth(40 * np.pi * 1/1400 * x + 0, 0.5) + 10
        #triangle = 10 * signal.sawtooth(2 * np.pi * 1/70 * x + 0.001, 0.5) + 10  # BJD: last used here 28.12.2020
        #trapezoid = trapezoid_signal(x, width=40, slope=5, amp=50)
        #piecewise = f(x)
        piecewise_choice = f3(x)
        #piecewise_1 = list(map(f, x))
        #trapezoid = trapezoid_signal(x, width=40, slope=10, amp=50)
        #composite = composite_triangle(x)
        #gradient = ( (t/3) - 8 ) * 0.0003 # (1+np.tanh(t-30)) * 0.0003
        #print("x = ", x)
        #return -np.exp(-0.5*(x*x+y*y))* center + (x+8) * gradient
        #u = x/25
        #u = x/10
        return -(
            np.exp(-0.5*( (x)**2 + y*y) ) #+ np.exp(-0.5*((x+8)**2 + y*y)) # 2 particles 16 units apart
            #np.exp(-0.5*((x+50)**2 + y*y)) #+ np.exp(-0.5*((x)**2 + y*y))
        ) * center # + piecewise_choice * gradient   # piecewise function --- G gap in G wall 13.3.2021
#......................................................

    source_functions = {
        'G': source_G,
    }

    flow = [0*x, 0*x]

    fluid_model_g = FluidModelG(
        x*0,
        x*0,
        x*0,
        flow,
        dx,
        dt=args.dt,
        params=args.model_params,
        source_functions=source_functions,
    )

    print("Rendering 'Nucleation and Motion in G gradient in 2D'")
    print("Lattice constant dx = {}, time step dt = {}".format(fluid_model_g.dx, fluid_model_g.dt))
    min_G = -4.672736908320116
    max_G = 0.028719261862332906
    min_X = -3.8935243721220334
    max_X = 1.2854028081816122
    min_Y = -0.7454193158963579
    max_Y = 4.20524950766914
    for n in progressbar.progressbar(range(args.num_frames)):
        fluid_model_g.step()
        if n % args.oversampling == 0:
            rgb = [
                6*(-fluid_model_g.G + max_G) / (max_G - min_G),
                5*(fluid_model_g.Y - min_Y) / (max_Y - min_Y),
                0.7*(fluid_model_g.X - min_X) / (max_X - min_X),
            ]
            zero_line = 1 - tf.exp(-600 * fluid_model_g.Y**2)
            frame = make_video_frame([c * zero_line for c in rgb])
            writer.append_data(frame)
#-------------------------BJD 28.1.2021----rough code at present-----------------------------------------        
            if n == 600:
                print("n = ", n)
                break
            #c1 = c1 + 1
            #print("H E L L O")
            #y1 = np.loadtxt("/home/brendan/software/tf2-model-g/arrays/quiver_array14/u.txt") #, delimiter=" :-) ", usecols=(120))  # (426, 240)
            #y2 = np.loadtxt("/home/brendan/software/tf2-model-g/arrays/quiver_array14/v.txt") #, delimiter=" :-) ", usecols=(120))  # (426, 240)

            c1 = c1 + 1
            # Set up grid and test data
            #nx, ny = 256, 1024
            #nx, ny = 240, 426
            #nx, ny = 426, 240  # changed round to check!
            nx, ny = 240, 426
            #(426,240)
            x1 = range(nx)
            y1 = range(ny)

            #data = numpy.random.random((nx, ny))
            U = np.loadtxt("/home/brendan/runs/tf2-model-g_1/tf2-model-g/arrays/quiver_array30/u.txt") #, delimiter=" :-) ", usecols=(120))  # (426, 240)
            V = np.loadtxt("/home/brendan/runs/tf2-model-g_1/tf2-model-g/arrays/quiver_array30/v.txt") #, delimiter=" :-) ", usecols=(120))  # (426, 240)
            # BJD 28.1.2021: u and v are saved as arrays in fluid_model_g.py as fllows:
            # np.savetxt("/home/brendan/software/tf2-model-g/arrays/quiver_array18/u.txt", self.u)
            # np.savetxt("/home/brendan/software/tf2-model-g/arrays/quiver_array18/v.txt", self.v)

            #data = np.loadtxt("/home/brendan/software/tf2-model-g/arrays/array12/Y.txt")

            #hf = plt.figure(figsize=(10,10))
            #ha = hf.add_subplot(111, projection='3d')
            #ha = hf.add_subplot(111, projection='2d')

            X1, Y1 = np.meshgrid(x1, y1)  # `plot_surface` expects `x` and `y` data to be 2D
            #ha.plot_surface(X, Y, data)
            #ha.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=cm.coolwarm,
            #                       linewidth=0, antialiased=False)
            #ha.plot_surface(X.T, Y.T, data)

            #surf = ha.plot_surface(X1, Y1, data, rstride=1, cstride=1, cmap=cm.coolwarm,
            #           linewidth=0, antialiased=False)
            #ha.set_zlim(-1, 3)
            #hf.colorbar(surf, shrink=0.5, aspect=10)
            #hf = plt.figure(figsize=(10,10))
            """
            fig, hf = plt.subplots(figsize=(10,10))
            #ax3.set_title("pivot='tip'; scales with x view")
            hf.set_title("pivot='tip'; scales with x view" + str(c1))
            M = np.hypot(U, V)
            Q = hf.quiver(X1, Y1, U, V, M, units='x', pivot='tip', width=0.022,
               scale=1 / 0.15)
            qk = hf.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
            hf.scatter(X1, Y1, color='0.5', s=1)
            """
            fig, hf = plt.subplots(figsize=(10,10))
            hf.set_title("pivot='mid'; every 1th arrow; units='velocity vector'" + str(c1))
            Q = hf.quiver(X1[::5, ::5], Y1[::5, ::5], U[::5, ::5], V[::5, ::5],
               pivot='mid', units='inches')
            qk = hf.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
            hf.scatter(X1[::5, ::5], Y1[::5, ::5], color='r', s=5)

            plt.savefig('/home/brendan/runs/tf2-model-g_1/tf2-model-g/plots/2D_video36/2D_video_velocity_' + str(c1) + '.png')
#===========================================================================
    #     max_G = max(max_G, tf.reduce_max(fluid_model_g.G).numpy())
    #     min_G = min(min_G, tf.reduce_min(fluid_model_g.G).numpy())
    #     max_X = max(max_X, tf.reduce_max(fluid_model_g.X).numpy())
    #     min_X = min(min_X, tf.reduce_min(fluid_model_g.X).numpy())
    #     max_Y = max(max_Y, tf.reduce_max(fluid_model_g.Y).numpy())
    #     min_Y = min(min_Y, tf.reduce_min(fluid_model_g.Y).numpy())

    # print(min_G, max_G, min_X, max_X, min_Y, max_Y)


def charged_nucleation_in_2D(writer, args, R=30, D=25, weights=(0, -10, -8, 8)):
    dx = 2*R / args.height
    x = (np.arange(args.width) - args.width // 2) * dx
    y = (np.arange(args.height) - args.height // 2) * dx
    x, y = np.meshgrid(x, y, indexing='ij')

    def source_G(t):
        amount = np.exp(-0.5*(t-5)**2)
        return (
            np.exp(-0.5*((x-D)**2+y*y)) * weights[0] +
            np.exp(-0.5*((x+D)**2+y*y)) * weights[1]
        ) * amount

    def source_X(t):
        amount = np.exp(-0.5*(t-5)**2)
        return (
            np.exp(-0.5*((x-D)**2+y*y)) * weights[2] +
            np.exp(-0.5*((x+D)**2+y*y)) * weights[3]
        ) * amount

    source_functions = {
        'G': source_G,
        'X': source_X,
    }

    noise_scale = 1e-4
    model_g = ModelG(
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale,
        dx,
        dt=args.dt,
        params=args.model_params,
        source_functions=source_functions,
    )

    print("Rendering 'Charged nucleation in 2D'")
    print("Lattice constant dx = {}, time step dt = {}".format(model_g.dx, model_g.dt))
    min_G = -4.672736908320116
    max_G = 0.028719261862332906
    min_X = -3.8935243721220334
    max_X = 1.2854028081816122
    min_Y = -0.7454193158963579
    max_Y = 4.20524950766914
    for n in progressbar.progressbar(range(args.num_frames)):
        model_g.step()
        if n % args.oversampling == 0:
            rgb = [
                6*(-model_g.G + max_G) / (max_G - min_G),
                5*(model_g.Y - min_Y) / (max_Y - min_Y),
                0.7*(model_g.X - min_X) / (max_X - min_X),
            ]
            zero_line = 1 - tf.exp(-600 * model_g.Y**2)
            frame = make_video_frame([c * zero_line for c in rgb])
            writer.append_data(frame)

def soliton_in_g_well_2D(writer, args, R=25, D=15):
    dx = 2*R / args.height
    x = (np.arange(args.width) - args.width // 2) * dx
    y = (np.arange(args.height) - args.height // 2) * dx
    x, y = np.meshgrid(x, y, indexing='ij')

    def source_G(t):
        nucleator = -np.exp(-0.5*(t-5)**2)
        #potential = 0.015 * (np.tanh(t-25) + 1)
        #potential = 0.0003 * (np.tanh(t-25) + 1)   # gradient = (1+np.tanh(t-30)) * 0.0003 # see above

        #u = x / R
        #u = x/10 - 5  # see: "soliton_in_g_well_2D___1_seed__gradient_1__2a.mp4"
        #u = x/15 # see: "soliton_in_g_well_2D___1_seed__gradient_2__2b.mp4"
        #u = x/25
        return (
            #np.exp(-0.5*((x-D)**2+y*y)) * nucleator +
            #(u*u - 1) * potential
            np.exp( -0.5*((x)**2 + y*y) ) * nucleator #+ (x+8) * potential
            #(u*u - 1) * potential #+ (x+8) * gradient
        )

    source_functions = {
        'G': source_G,
    }

    noise_scale = 1e-4
    model_g = ModelG(
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale,
        dx,
        dt=args.dt,
        params=args.model_params,
        source_functions=source_functions,
    )

    print("Rendering 'Soliton in G-well in 2D'")
    print("Lattice constant dx = {}, time step dt = {}".format(model_g.dx, model_g.dt))
    G_scale = 0.02
    X_scale = 0.25
    Y_scale = 0.5
    for n in progressbar.progressbar(range(args.num_frames)):
        model_g.step()
        if n % args.oversampling == 0:
            rgb = [
                (G_scale*0.5 - model_g.G) / G_scale,
                (model_g.Y - Y_scale*0.5) / Y_scale,
                (model_g.X - X_scale*0.5) / X_scale,
            ]
            zero_line = 1 - tf.exp(-600 * model_g.Y**2)
            frame = make_video_frame([c * zero_line for c in rgb])
            writer.append_data(frame)

# TODO: Requires some work. Unstable like this.
def nucleation_3D(writer, args, R=20):
#    raise NotImplementedError("Needs some work")
    params = {
        "A": 3.4,
        "B": 13.5,
        "k2": 1.0,
        "k-2": 0.1,
        "k5": 0.9,
        "D_G": 1.0,
        "D_X": 1.0,
        "D_Y": 1.95,
        "density_G": 1.0,
        "density_X": 0.0002,
        "density_Y": 0.043,
        "base-density": 9.0,
        "viscosity": 0.3,
        "speed-of-sound": 1.0,
    }

    dx = 2*R / args.height
    x = (np.arange(args.width) - args.width // 2) * dx
    y = (np.arange(args.height) - args.height // 2) * dx
    z = y
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    def source_G(t):
        center = np.exp(-0.3*(t-6)**2) * 10
        return -np.exp(-0.5*(x*x+y*y+z*z)) * center

    source_functions = {
        'G': source_G,
    }

    # We need some noise to break spherical symmetry
    noise_scale = 1e-4
    G = bl_noise(x.shape) * noise_scale
    X = bl_noise(x.shape) * noise_scale
    Y = bl_noise(x.shape) * noise_scale
    flow = [
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale,
        bl_noise(x.shape) * noise_scale
    ]

    fluid_model_g = FluidModelG(
        G, X, Y,
        flow,
        dx,
        dt=args.dt,
        params=params,
        source_functions=source_functions,
    )

    flow_particle_origins = []
    for _ in range(1000):
        flow_particle_origins.append([np.random.rand() * s for s in x.shape])

    flow_particles = tf.constant(flow_particle_origins, dtype='float64')
    flow_streaks = 0*x[:,:,0]

    print("Rendering 'Nucleation and Motion in G gradient in 3D'")
    print("Lattice constant dx = {}, time step dt = {}".format(fluid_model_g.dx, fluid_model_g.dt))
    for n in progressbar.progressbar(range(args.num_frames)):
        fluid_model_g.step()
        for _ in range(20):
            indices = tf.cast(flow_particles, 'int32')
            for index in indices.numpy():
                flow_streaks[index[0], index[1]] += 0.15 / args.oversampling
            dx = tf.gather_nd(fluid_model_g.u, indices)
            dy = tf.gather_nd(fluid_model_g.v, indices)
            dz = tf.gather_nd(fluid_model_g.w, indices)
            flow_particles = (flow_particles + tf.stack([dx, dy, dz], axis=1) * 400) % x.shape
        if n % args.oversampling == 0:
            rgb = [
                tf.reduce_mean((7*fluid_model_g.G)**2, axis=2) + flow_streaks,
                tf.reduce_mean((4*fluid_model_g.Y)**2, axis=2),
                tf.reduce_mean((2*fluid_model_g.X)**2, axis=2),
            ]
            frame = make_video_frame(rgb)
            writer.append_data(frame)
            flow_streaks *= 0
            flow_particles = tf.constant(flow_particle_origins, dtype='float64')

if __name__ == '__main__':
    episodes = {
        'nucleation_and_motion_in_fluid_2D': nucleation_and_motion_in_G_gradient_fluid_2D,
        'charged_nucleation_in_2D': charged_nucleation_in_2D,
        'soliton_in_g_well_2D': soliton_in_g_well_2D,
        'nucleation_3D': nucleation_3D,
    }

    parser = argparse.ArgumentParser(description='Render audio samples')
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--params', type=str, help='Parameter YAML file name')
    parser.add_argument('--episode', choices=episodes.keys())
    parser.add_argument('--resolution', choices=RESOLUTIONS.keys(), help='Video and simulation grid resolution')
    parser.add_argument('--width', type=int, help='Video and simulation grid width', metavar='W')
    parser.add_argument('--height', type=int, help='Video and simulation grid height', metavar='H')
    parser.add_argument('--framerate', type=int, help='Video frame rate')
    parser.add_argument('--oversampling', type=int, help='Add extra simulation time steps between video frames for stability')
    parser.add_argument('--video-quality', type=int, help='Video quality factor')
    parser.add_argument('--video-duration', type=float, help='Duration of video to render in seconds')
    parser.add_argument('--simulation-duration', type=float, help='Amount of simulation to run')
    args = parser.parse_args()

    args.model_params = {}
    if args.params:
        with open(args.params) as f:
            params = yaml.load(f, Loader=Loader)
            for key, value in params.items():
                if not getattr(args, key):
                    setattr(args, key, value)

    if not args.episode:
        raise ValueError("Missing episode argument. Must be present in either parameter YAML file or as a program argument.")

    if not args.framerate:
        args.framerate = 24
    if not args.oversampling:
        args.oversampling = 1
    if not args.video_quality:
        args.video_quality = 10

    writer = imageio.get_writer(args.outfile, fps=args.framerate, quality=args.video_quality, macro_block_size=1)

    # Compute derived parameters
    if args.resolution:
        width, height = RESOLUTIONS[args.resolution]
        if not args.width:
            args.width = width
        if not args.height:
            args.height = height
    if (not args.width) or (not args.height):
        raise ValueError("Invalid or missing resolution")
    args.aspect = args.width / args.height
    args.num_frames = int(args.video_duration * args.oversampling * args.framerate)
    args.dt = args.simulation_duration / args.num_frames

    episodes[args.episode](writer, args)
    writer.close()
#=======================BJD make video from .png files 24.11.2020===========================
def save1():
    #os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")
    os.system("ffmpeg -r 1 -i /home/brendan/runs/tf2-model-g_1/tf2-model-g/plots/2D_video36/2D_video_velocity_%01d.png -vcodec mpeg4 -y nucleation_and_motion_in_fluid_2D___velocity_vector___2d__nx_ny_inverted__video_150.mp4")

save1()

#============================================================================================
