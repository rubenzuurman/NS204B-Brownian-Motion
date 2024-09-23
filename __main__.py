import os

from src.analysis import get_trajectories, analyse_trajectories
from src.data_loader import load_video, load_batch_cache, load_link_cache, generate_batch_data, generate_link_data, save_batch_cache, save_link_cache

def load_data(video_path: str, number_of_frames: int, force_regenerate: bool=False):
    # Get video filename.
    video_filename = os.path.basename(video_path)
    
    # Load data.
    frames = load_video(video_path)
    if force_regenerate:
        batch_data = generate_batch_data(frames, number_of_frames, particle_diameter=9, minimum_mass=100)
        link_data  = generate_link_data(batch_data)
    else:
        batch_success, batch_data = load_batch_cache(video_filename)
        link_success, link_data   = load_link_cache(video_filename)
        if not batch_success:
            batch_data = generate_batch_data(frames, number_of_frames, particle_diameter=9, minimum_mass=100)
            link_data  = generate_link_data(batch_data)
        elif not link_success:
            link_data  = generate_link_data(batch_data)
    
    # Save cache.
    save_batch_cache(batch_data, video_filename)
    save_link_cache(link_data, video_filename)
    
    # Return data.
    return frames, batch_data, link_data

def main():
    # Set video path.
    video_path = "data/metingen/005_mass_percent_500nm_40x_3.wmv"
    
    # Set number of frames to analyse.
    number_of_frames = 30
    
    # Load data.
    frames, batch_data, link_data = load_data(video_path, number_of_frames, force_regenerate=True)
    
    # Generate trajectories.
    trajectory_data = get_trajectories(frames, batch_data, link_data)
    
    # Calculate diffusion constant.
    diffusion_constant = analyse_trajectories(trajectory_data, number_of_frames=number_of_frames)

if __name__ == "__main__":
    main()
