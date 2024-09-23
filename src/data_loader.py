import os

import cv2
from loguru import logger
import matplotlib.pyplot as plt
import trackpy as tp

from src.cache import load_cache, save_cache

def load_video(path: str):
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
    
    # TODO: Remove this before handing in for grading (together with the other code for generating images).
    # Save last frame in grayscale as grayscale_frame.png.
    plt.imsave("grayscale_frame.png", image_grayscale, cmap="gray", vmin=0, vmax=255)
    logger.info(f"Saved frame number {len(frames) - 1} in grayscale to 'grayscale_frame.png'.")
    
    # Return list of frames.
    return frames

def load_batch_cache(video_filename: str):
    """
    Returns (True, data) if success, else returns (False, None).
    """
    path = os.path.join(os.getcwd(), "cache", "batch", f"{os.path.splitext(video_filename)[0]}.pickle")
    result = load_cache(path)
    return (not (result is None), result)

def load_link_cache(video_filename: str):
    """
    Returns (True, data) if success, else returns (False, None).
    """
    path = os.path.join(os.getcwd(), "cache", "link", f"{os.path.splitext(video_filename)[0]}.pickle")
    result = load_cache(path)
    return (not (result is None), result)

def save_batch_cache(data, video_filename: str):
    path = os.path.join(os.getcwd(), "cache", "batch", f"{os.path.splitext(video_filename)[0]}.pickle")
    cache_dir = os.path.dirname(path)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    save_cache(data, path)

def save_link_cache(data, video_filename: str):
    path = os.path.join(os.getcwd(), "cache", "link", f"{os.path.splitext(video_filename)[0]}.pickle")
    cache_dir = os.path.dirname(path)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    save_cache(data, path)

def generate_batch_data(frames, number_of_frames, particle_diameter, minimum_mass):
    """
    Number of frames: number of frames to analyse. If bigger than the total number of frames, the entire file will be analysed.
    Particle diameter: not sure what it does exactly, but increase if dip in subpixel bias hist.
    Minimum mass: minimum integrated brightness of a particle.
    """
    # Generate new batch data.
    logger.info(f"Generating new batch data (diameter={particle_diameter}, minmass={minimum_mass})...")
    batch_data = tp.batch(frames[:number_of_frames], diameter=particle_diameter, invert=True, minmass=minimum_mass, processes=1)
    
    # Return batch data.
    return batch_data

def generate_link_data(batch_data, search_range=5, memory=3):
    """
    Search range: Maximum distance a particle can travel between frames to be considered a valid trajectory.
    Memory:       Number of frames a particle can disappear for and reappear nearby to be considered the same particle.
    """
    # Suppress messages from trackpy during linking, since this step executes the fastest.
    tp.quiet()
    
    # Generate new link data.
    logger.info(f"Generating new link data (search_range={search_range}, memory={memory})...")
    link_data = tp.link(batch_data, search_range=search_range, memory=memory)
    
    # Return link data.
    return link_data
