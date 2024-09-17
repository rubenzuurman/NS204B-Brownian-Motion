from src.analysis import load_data, get_trajectories, analyse_trajectories

def main():
    # Set video path.
    video_path = "data/test_13_09_2024_05.wmv"
    
    # Set number of frames to analyse.
    number_of_frames = 1000
    
    # Load data.
    frames, batch_data, link_data = load_data(video_path, number_of_frames=number_of_frames)
    
    # Generate trajectories.
    trajectory_data = get_trajectories(frames, batch_data, link_data)
    
    # Calculate diffusion constant.
    diffusion_constant = analyse_trajectories(trajectory_data)

if __name__ == "__main__":
    main()
