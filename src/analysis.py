import os

import cv2
import matplotlib.pyplot as plt
import pims
import trackpy as tp

def load_data(path: str):
    # Check if path exists.
    assert os.path.isfile(path), f"File could not be found: '{path}'."
    
    # Check if filetype is valid.
    assert os.path.splitext(path)[1] == ".wmv", "Only WMV files are supported currently."
    
    # Load video handle using cv2.
    video_handle = cv2.VideoCapture(path)
    
    # Extract frames from video handle.
    frames = []
    while True:
        # Do something?
        video_handle.grab()
        
        # Get frame, it's a 3d np array with u8 pixel values (rgb).
        return_value, image = video_handle.retrieve()
        if not return_value:
            break
        
        # Convert frame to grayscale (e.g. get green channel).
        image_grayscale = image[:, :, 1]
        
        # Add frame to frames list.
        frames.append(image_grayscale)
    
    # Save image (uncomment line below to save last frame in grayscale as test.png).
    #plt.imsave("test.png", image_grayscale, cmap="gray", vmin=0, vmax=255)
    
    # Return list of frames.
    return frames

def preprocess(data):
    pass

def get_trajectories(frames):
    f = tp.locate(frames[0], 11, invert=True, minmass=600)
    
    fig, ax = plt.subplots(ncols=2, nrows=1)
    ax[0].hist(f["mass"], bins=20)
    ax[0].set(xlabel="mass", ylabel="count")
    
    tp.annotate(f, frames[0], ax=ax[1])
    
    fig.savefig("hist_and_annotation.png")
