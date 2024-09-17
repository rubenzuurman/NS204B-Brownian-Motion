import os

import cv2
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
# TODO: Maybe remove these before submitting code for grading.
pd.set_option("display.max_columns", None) # Make sure all dataframe columns are printed when printing dataframe
pd.set_option("display.max_rows", None) # Make sure all dataframe rows are printed when printing dataframe
pd.set_option("display.width", 200) # Set terminal width in number of characters to prevent new line in middle of printing dataframe
import pims
import trackpy as tp

from src.cache import cache_exists, load_cache, save_cache

def load_frames_from_video(path: str):
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
    
    # Return list of frames.
    return frames

def attempt_to_load_batch_cache():
    """
    Returns (False, None) if the cache does not exist or could not be loaded properly. Returns (True, data) otherwise, where data is the batch data (dataframe).
    """
    # Set batch cache path.
    batch_cache_path = "cache/batch_df.pickle"
    logger.info("Attempting to load batch data...")
    
    # Return false if the file does not exist.
    if not cache_exists(batch_cache_path):
        logger.warning("Batch cache file does not exist.")
        return (False, None)
    
    # Attempt to load batch data.
    batch_data = load_cache(batch_cache_path)
    if batch_data is None:
        logger.warning("Failed to load batch data from cache file.")
        return (False, None)
    
    # Return data if everything succeeded.
    logger.success("Loaded batch data from cache.")
    return (True, batch_data)

def attempt_to_load_link_cache():
    """
    Returns (False, None) if the cache does not exist or could not be loaded properly. Returns (True, data) otherwise, where data is the link data (dataframe).
    Note: If the batch data is regenerated, the link data should be regenerated as well, since the batch data is not guaranteed to be identical to the last.
    """
    # Set link cache path.
    link_cache_path = "cache/link_df.pickle"
    logger.info("Attempting to load link data...")
    
    # Return false if the file does not exist.
    if not cache_exists(link_cache_path):
        logger.warning("Link cache file does not exist.")
        return (False, None)
    
    # Attempt to load link data.
    link_data = load_cache(link_cache_path)
    if link_data is None:
        logger.warning("Failed to load link data from cache file.")
        return (False, None)
    
    # Return data if everything succeeded.
    logger.success("Loaded link data from cache.")
    return (True, link_data)

def generate_batch_data(frames, number_of_frames):
    # Generate new batch data.
    logger.info("Generating new batch data...")
    batch_data = tp.batch(frames[:number_of_frames], 11, invert=True, minmass=600, processes=1)
    
    # Save batch data.
    batch_cache_path = "cache/batch_df.pickle"
    save_cache(data, batch_cache_path)
    
    # Return batch data.
    return batch_data

def generate_link_data(batch_data):
    # Suppress messages from trackpy during linking, since this step executes the fastest.
    tp.quiet()
    
    # Generate new link data.
    logger.info("Generating new link data...")
    link_data = tp.link(batch_data, 5, memory=3)
    
    # Save link data.
    link_cache_path = "cache/link_df.pickle"
    save_cache(data, link_cache_path)
    
    # Return link data.
    return link_data

def load_data(path: str, number_of_frames: int):
    # Load frames from video.
    logger.info(f"Loading frames from file '{path}'...")
    frames = load_frames_from_video(path)
    
    # Load batch data.
    batch_success, batch_data = attempt_to_load_batch_cache()
    link_success, link_data = attempt_to_load_link_cache()
    if not batch_success:
        # Generate batch data if batch cache failed to load.
        batch_data = generate_batch_data(frames, number_of_frames)
        
        # Generate link data if batch cache failed to load, since the batch data may not be identical to the last.
        link_data = generate_link_data(batch_data)
    if batch_success and not link_success:
        # Generate link data if link cache failed to load.
        link_data = generate_link_data(batch_data)
    
    # Return frames and processed data.
    return (frames, batch_data, link_data)

def get_trajectories(frames, batch_data, link_data):
    # Plot batch data.
    fig, ax = plt.subplots(ncols=3, nrows=1)
    
    ax[0].hist(batch_data["mass"], bins=20)
    ax[0].set(xlabel="mass", ylabel="count")
    
    tp.annotate(batch_data, frames[0], ax=ax[1])
    
    #print(tp.subpx_bias(batch_data)[0][0])
    subpx_bias(batch_data, ax=ax[2])
    
    fig.savefig("hist_and_annotation.png")
    
    # Filter stubs, e.g. remove trajectories shorter than some number of frames (25 in this case).
    t_filtered = tp.filter_stubs(link_data, 25)
    
    # Print to see how many trajectories were removed.
    print("Before:", link_data["particle"].nunique())
    print("After: ", t_filtered["particle"].nunique())
    
    # Plot trjactory data.
    fig, ax = plt.subplots(ncols=1, nrows=1)
    tp.mass_size(t_filtered.groupby("particle").mean(), ax=ax)
    fig.savefig("mass_size.png")
    
    # At this stage it would be good to filter the trajectory dataframe for e.g. low mass, high size, or high eccentricity.
    #t_filtered_pruned = t_filtered[((t_filtered['mass'] > 50) & (t_filtered['size'] < 2.6) & (t_filtered['ecc'] < 0.3))]
    t_filtered_pruned = t_filtered
    
    # Plot trajectories.
    fig, ax = plt.subplots(ncols=1, nrows=1)
    tp.plot_traj(t_filtered_pruned, ax=ax)
    fig.savefig("trajectories.png")

def subpx_bias(f, pos_columns=None, ax=None):
    # Copied from the source of trackpy, may be deleted when no longer needed.
    # TODO: Remove this function before handing in for grading.
    """Histogram the fractional part of the x and y position.

    Parameters
    ----------
    f : DataFrame
    pos_columns : list of column names, optional

    Notes
    -----
    If subpixel accuracy is good, this should be flat. If it depressed in the
    middle, try using a larger value for feature diameter."""
    if pos_columns is None:
        if 'z' in f:
            pos_columns = ['x', 'y', 'z']
        else:
            pos_columns = ['x', 'y']
    axlist = f[pos_columns].applymap(lambda x: x % 1).hist(ax=ax)
    return axlist
