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
    # Set number of frames to analyse.
    number_of_frames = 600
    
    # Load batch data.
    batch_cache_path = "cache/batch_df.pickle"
    logger.info("Loading batch data...")
    if cache_exists(batch_cache_path):
        # Attempt to load cache if it exists.
        f = load_cache(batch_cache_path)
        
        # If cache fails to load, generate new data and attempt to overwrite it.
        # NOTE: The processes=1 option was EXTREMELY important in my case, if not present it used a ton of RAM and made all my open chrome windows crash.
        if f is None:
            logger.warning("Failed to load batch data from cache, generating new data...")
            f = tp.batch(frames[:number_of_frames], 11, invert=True, minmass=600, processes=1)
            save_cache(f, batch_cache_path)
            logger.warning("Generated new data and saved to cache.")
        else:
            logger.success("Loaded batch data from cache.")
    else:
        # Generate data and save to cache if cache does not exist.
        logger.warning("Generating new data...")
        f = tp.batch(frames[:number_of_frames], 11, invert=True, minmass=600, processes=1)
        save_cache(f, batch_cache_path)
        logger.warning("Generated new data and saved to cache.")
    
    # Plot batch data.
    fig, ax = plt.subplots(ncols=3, nrows=1)
    
    ax[0].hist(f["mass"], bins=20)
    ax[0].set(xlabel="mass", ylabel="count")
    
    tp.annotate(f, frames[0], ax=ax[1])
    
    #print(tp.subpx_bias(f)[0][0])
    subpx_bias(f, ax=ax[2])
    
    fig.savefig("hist_and_annotation.png")
    
    # Suppress messages from trackpy during linking, since this step executes the fastest.
    tp.quiet()
    
    # Load link data.
    link_cache_path = "cache/link_df.pickle"
    logger.info("Loading link data...")
    if cache_exists(link_cache_path):
        # Attempt to load cache if it exists.
        t = load_cache(link_cache_path)
        
        # If cache fails to load, generate new data and attempt to overwrite it.
        if t is None:
            t = tp.link(f, 5, memory=3)
            save_cache(t, link_cache_path)
            logger.warning("Failed to load link data from cache, generated new data and saved to cache.")
        else:
            logger.success("Loaded link data from cache.")
    else:
        # Generate data and save to cache if cache does not exist.
        t = tp.link(f, 5, memory=3)
        save_cache(t, link_cache_path)
        logger.warning("Generated new data and saved to cache.")
    
    # Filter stubs, e.g. remove trajectories shorter than some number of frames.
    t_filtered = tp.filter_stubs(t, 25)
    
    # Print to see how many trajectories were removed.
    print("Before:", t["particle"].nunique())
    print("After:", t_filtered["particle"].nunique())
    
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
