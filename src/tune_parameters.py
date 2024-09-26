import random as rnd

from loguru import logger
import matplotlib.pyplot as plt
import trackpy as tp

def show_tune(frames, video_filename, particle_diameter, minimum_mass):
    # Generate 5 random numbers.
    rnd.seed(video_filename)
    
    # Get 5 random frames.
    frame_numbers = [rnd.randint(0, len(frames) - 1) for _ in range(5)]
    useful_frames = []
    for n in frame_numbers:
        useful_frames.append(frames[n])
    
    # Get batch data on those frames.
    batch_data = tp.batch(useful_frames, diameter=particle_diameter, invert=True, minmass=minimum_mass, processes=1)
    
    # Initialize plot.
    fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(50, 32), gridspec_kw={"height_ratios": [1, 1, 1]})
    
    # Annotate each frame into plot.
    for index in range(len(frame_numbers)):
        # Plot annotated frames on row 0.
        tp.annotate(batch_data[batch_data["frame"] == index], useful_frames[index], ax=axs[0][index])
        axs[0][index].set_title(f"Frame {frame_numbers[index]}")
        
        # Plot pixel biases on row 1.
        pixel_biases = batch_data[batch_data["frame"] == index][["x", "y"]].map(lambda x: x % 1)
        pixel_biases_x = pixel_biases["x"]
        pixel_biases_y = pixel_biases["y"] + 1
        axs[1][index].hist(pixel_biases_x)
        axs[1][index].hist(pixel_biases_y)
        axs[1][index].legend(["x", "y"])
        
        # Plot mass histogram on row 2.
        #axs[2][index].set_title(f"Histogram of particles in frame {frame_no}\nbinned by mass")
        axs[2][index].hist(batch_data[batch_data["frame"] == index]["mass"], bins=20)
        axs[2][index].set(xlabel="Mass", ylabel="Count")
    
    mass_percent = int(video_filename[:3]) / 10
    mass_percent, _, _, particle_size, magnification, sample_id, recording_id = video_filename[:-4].split("_")
    fig.suptitle(f"{video_filename}\n{particle_size} at {magnification}\nSample {sample_id}{recording_id}", fontsize=20)
    fig.tight_layout()
    fig.savefig("tune_image.png")
    logger.info("Saved image to help tune parameters as 'tune_image.png'.")
