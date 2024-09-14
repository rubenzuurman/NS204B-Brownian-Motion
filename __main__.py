from src.analysis import load_data, get_trajectories

def main():
    frames = load_data("data/test_13_09_2024_05.wmv")
    trajectories = get_trajectories(frames)

if __name__ == "__main__":
    main()
