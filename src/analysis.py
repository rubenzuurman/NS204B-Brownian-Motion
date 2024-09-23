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

PARTICLE_DIAMETER = 15 # Particle diameter in pixels to look for (?)
MINIMUM_MASS = 1500     # Minimum mass of the particle to be included (read integrated brightness)
MINIMUM_TRAJECTORY_LENGTH_FRAMES = 25 # Trajectories with a length of less than 25 frames will be pruned

# TODO: Add text to axes in plots.
# TODO: Make printing of before and after trajectories more pretty.

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
    logger.info(f"Saved frame number {len(frames) - 1} in grayscale to 'grayscale_frame.png'.")
    
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
    logger.info(f"Generating new batch data (diameter={PARTICLE_DIAMETER}, minmass={MINIMUM_MASS})...")
    batch_data = tp.batch(frames[:number_of_frames], diameter=PARTICLE_DIAMETER, invert=True, minmass=MINIMUM_MASS, processes=1)
    
    # Save batch data.
    batch_cache_path = "cache/batch_df.pickle"
    save_cache(batch_data, batch_cache_path)
    
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
    
    # Save link data.
    link_cache_path = "cache/link_df.pickle"
    save_cache(link_data, link_cache_path)
    
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
    fig, ax = plt.subplots(ncols=2, nrows=1)
    
    frame_no = 40
    ax[0].set_title(f"Histogram of particles in frame {frame_no}\nbinned by mass")
    ax[0].hist(batch_data[batch_data["frame"] == frame_no]["mass"], bins=20)
    ax[0].set(xlabel="Mass", ylabel="Count")
    
    ax[1].set_title(f"Annotation of particles in frame {frame_no}")
    tp.annotate(batch_data[batch_data["frame"] == frame_no], frames[frame_no], ax=ax[1])
    
    # TODO: Find a better name for this plot.
    fig.tight_layout()
    fig.savefig("hist_and_annotation.png")
    logger.info("Plotted mass vs count histogram and annotated frame (hist_and_annotation.png).")
    
    # Plot subpixel bias. If minmass is too low, this will show a dip according to the walkthrough.
    #print(tp.subpx_bias(batch_data)[0][0])
    fig, ax = plt.subplots(ncols=1, nrows=1)
    subpx_bias(batch_data, ax=ax)
    fig.savefig("subpixel_bias.png")
    logger.info("Plotted subpixel bias (subpixel_bias.png).")
    
    # Filter stubs, e.g. remove trajectories shorter than some number of frames (25 in the walkthrough, or whatever is set in the global variable).
    t_filtered = tp.filter_stubs(link_data, threshold=MINIMUM_TRAJECTORY_LENGTH_FRAMES)
    logger.info(f"Removed trajectories shorter than {MINIMUM_TRAJECTORY_LENGTH_FRAMES} frames.")
    
    # Print to see how many trajectories were removed.
    print("Before:", link_data["particle"].nunique())
    print("After: ", t_filtered["particle"].nunique())
    
    # Plot trjactory data.
    fig, ax = plt.subplots(ncols=1, nrows=1)
    tp.mass_size(t_filtered.groupby("particle").mean(), ax=ax)
    fig.savefig("mass_vs_size.png")
    logger.info("Plotted mass vs size of all particles (mass_vs_size.png).")
    
    # TODO: At this stage it would be good to filter the trajectory dataframe for e.g. low mass, high size, or high eccentricity.
    #t_filtered_pruned = t_filtered[((t_filtered['mass'] > 50) & (t_filtered['size'] < 2.6) & (t_filtered['ecc'] < 0.3))]
    t_filtered_pruned = t_filtered
    
    # Plot trajectories.
    fig, ax = plt.subplots(ncols=1, nrows=1)
    tp.plot_traj(t_filtered_pruned, ax=ax)
    fig.savefig("trajectories.png")
    logger.info("Plotted raw particle trajectories (trajectories.png).")
    
    # Compute drift.
    drift = tp.compute_drift(t_filtered_pruned)
    
    # Plot drift.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(drift.index, drift["x"])
    ax.plot(drift.index, drift["y"])
    fig.savefig("drift.png")
    logger.info("Plotted drift (drift.png).")
    
    # Subtract drift from data.
    t_no_drift = tp.subtract_drift(t_filtered_pruned.copy(), drift)
    
    # Plot trajectories again.
    fig, ax = plt.subplots(ncols=1, nrows=1)
    tp.plot_traj(t_no_drift, ax=ax)
    fig.savefig("trajectories_no_drift.png")
    logger.info("Plotted particle trajectories after subtracting drift (trajectories_no_drift.png).")
    
    return t_no_drift

def analyse_trajectories(trajectory_data, number_of_frames):
    # Calculate mean squared displacement.
    microns_per_pixel = 1 / 6 # TODO: Measure this using the special 10x100 um slides.
    frames_per_second = 20
    
    del trajectory_data["particle"]
    msd = tp.imsd(trajectory_data, mpp=microns_per_pixel, fps=frames_per_second, max_lagtime=number_of_frames)
    logger.info(f"Calculated mean squared displacement of all particles (mpp={microns_per_pixel}, fps={frames_per_second}).")
    
    # Plot msd vs time.
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(msd.index, msd, "k-", alpha=0.1)  # black lines, semitransparent
    ax.set(xlabel="lag time $t$", ylabel=r"$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig("msd_vs_lag_time_log.png")
    logger.info("Plotted mean squared displacement of all particles vs time (log) (msd_vs_lag_time_log.png).")
    
    # Plot particle with index 0 msd vs time.
    time = list(msd.index)
    fig, ax = plt.subplots(ncols=1, nrows=1)
    if len(msd.columns) > 0:
        particle0 = list(msd[msd.columns[0]])
        ax.plot(time, particle0)
        logger.info(f"Plotted mean squared displacement of particle with index {msd.columns[0]} vs time (particle0_msd_vs_time.png).")
    else:
        logger.warning(f"Failed to plot mean squared displacement of the first available particle. Reason: No particles were available for msd calculation. (This *probably* should have crashed when the tp.link() function was called.)") # TODO: Check if the tp.link() function always fails when there were no particle found by the tp.batch() function, or just remove the 'Reason' part of the above log message and everything after it.
    fig.savefig("particle0_msd_vs_time.png")
    
    # Calculate (and plot) cumulative distance travelled by each particle.
    cumulative_distance = [[sum(list(msd[particle_no])[0:index]) for index in range(len(msd[particle_no]))] for particle_no in msd.columns]
    fig, ax = plt.subplots(ncols=1, nrows=1)
    for particle_cumul_list in cumulative_distance:
        ax.plot(time, particle_cumul_list)
    fig.savefig("all_particles_cumulative_distance_vs_time.png")
    logger.info("Plotted cumulative distance travelled of all particles vs time (all_particles_cumulative_distance_vs_time.png).")
    
    # Print time and particle data to use in mean and standard deviation calculation.
    print(f"time = {time}")
    print(f"particle_distancement = {cumulative_distance}")

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
    axlist = f[pos_columns].map(lambda x: x % 1).hist(ax=ax)
    return axlist
