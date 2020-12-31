import os # BJD added 24.11.2020 - for make video

def save1():
    #os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")
    os.system("ffmpeg -r 1 -i /home/brendan/software/tf2-model-g/plots/3D_video18/3D_video_XYG_%01d.png -vcodec mpeg4 -y nucleation_and_motion_in_fluid_2Dsurf_in_3D___X_seed___gradient__video_27.mp4")

save1()
