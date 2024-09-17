from src.analysis import load_data, get_trajectories

def main():
    # Set video path.
    video_path = "data/test_13_09_2024_05.wmv"
    
    # Set number of frames to analyse.
    number_of_frames = 600
    
    # Load data.
    frames, batch_data, link_data = load_data(video_path, number_of_frames=number_of_frames)
    
    # Generate trajectories.
    trajectories = get_trajectories(frames, batch_data, link_data)

if __name__ == "__main__":
    main()
