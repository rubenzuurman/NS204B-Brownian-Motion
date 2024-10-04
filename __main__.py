import os
import pickle
import sys

from loguru import logger

from src.analysis import get_trajectories, analyse_trajectories
from src.data_loader import load_video, load_batch_cache, load_link_cache, generate_batch_data, generate_link_data, save_batch_cache, save_link_cache
import src.tune_parameters as tune_parameters

def load_data(video_path: str, cmd_args):
    # Get video filename.
    video_filename = os.path.basename(video_path)
    
    # Load data.
    frames = load_video(video_path)
    if cmd_args["force_regenerate"]:
        batch_data = generate_batch_data(frames, cmd_args["num_frames"], particle_diameter=cmd_args["particle_diameter"], minimum_mass=cmd_args["minmass"])
        link_data  = generate_link_data(batch_data)
    else:
        batch_success, batch_data = load_batch_cache(video_filename)
        link_success, link_data   = load_link_cache(video_filename)
        if not batch_success:
            batch_data = generate_batch_data(frames, cmd_args["num_frames"], particle_diameter=cmd_args["particle_diameter"], minimum_mass=cmd_args["minmass"])
            link_data  = generate_link_data(batch_data)
        elif not link_success:
            link_data  = generate_link_data(batch_data)
    
    # Save cache.
    save_batch_cache(batch_data, video_filename)
    save_link_cache(link_data, video_filename)
    
    # Return data.
    return frames, batch_data, link_data

def print_usage():
    print("Usage:")
    print("    tune=0 -> run entire analysis")
    print("    tune=1 -> run annotate and such on 5 random frames (saved to tune_image.png)")
    print("    particle_diameter parameter (positive odd integer) can be used to set particle diameter")
    print("    minmass parameter (positive integer) can be used to set minimum mass")
    print("    max_particle_size parameter (nonnegative float) can be used to set maximum particle size after analysing trajectories (I'm not exactly sure what this does, but I added it since mass can be smeared out, and I don't really wanna track smeared-out particles) (mass_vs_size.png can be used after analysing the video to determine if this parameter needs to be altered)")
    print("    num_frames parameter (positive integer or \"max\" no quotations) can be used to set the number of frames to analyse (when tune=0, the number of frames will always be equal to the filter_stubs argument)")
    print("    search_range parameter (nonnegative integer) can be used to set the maximum distance a particle can travel between frames to be considered a valid trajectory")
    print("    memory parameter (nonnegative integer) can be used to set the number of frames a particle can disappear for and reappear nearby to be considered the same particle")
    print("    min_traj_len parameter (nonnegative integer) can be used to set the minimum length in frames of particle trajectories to be included in the analysis")
    print("    force_regenerate parameter (0 or 1) can be used to ignore any cache that might exist and overwrite it")
    print("It's currently recommended to leave 'search_range', 'memory', and 'min_traj_len' on default (non supplied), since those values seem to work fine.")
    print("The parameters 'particle_diameter' and 'minmass' can be tuned by also supplying 'tune=1' and looking at the file 'tune_image.png'.")

def parse_cmd_args(args: list[str]):
    DEFAULT_TUNE_VALUE = False
    DEFAULT_PARTICLE_DIAMETER = 9
    DEFAULT_MINMASS = 0
    DEFAULT_MAX_PARTICLE_SIZE = 100.0
    DEFAULT_SEARCH_RANGE = 5
    DEFAULT_MEMORY = 3
    DEFAULT_MIN_TRAJ_LEN = 25
    DEFAULT_FORCE_REGENERATE = False
    
    tune_bool = False
    particle_diameter = 5
    error_encountered = False
    
    arguments_parsed = {}
    
    # NOTE: If error_encountered is True, this function will sys.exit(0) the program.
    for arg in args:
        name, value = arg.split("=")
        if name == "tune":
            try:
                tune_bool = int(value)
                if tune_bool == 0:
                    tune_bool = False
                else:
                    tune_bool = True
            except Exception as e:
                logger.critical(f"Could not parse 'tune' argument: '{e}'")
                print_usage()
                tune_bool = None
                error_encountered = True
            arguments_parsed["tune"] = tune_bool
        elif name == "particle_diameter":
            try:
                particle_diameter = int(value)
                if particle_diameter < 0 or particle_diameter % 2 == 0:
                    logger.critical("Particle diameter must be a positive odd integer.")
                    print_usage()
                    error_encountered = True
            except Exception as e:
                logger.critical(f"Could not parse 'particle_diameter' argument: '{e}'")
                print_usage()
                particle_diameter = None
                error_encountered = True
            arguments_parsed["particle_diameter"] = particle_diameter
        elif name == "minmass":
            try:
                minmass = int(value)
                if minmass < 0:
                    logger.critical("Minimum mass must be a nonnegative integer.")
                    print_usage()
                    error_encountered = True
            except Exception as e:
                logger.critical(f"Could not parse 'minmass' argument: '{e}'")
                print_usage()
                minmass = None
                error_encountered = True
            arguments_parsed["minmass"] = minmass
        elif name == "max_particle_size":
            try:
                max_particle_size = float(value)
                if max_particle_size < 0:
                    logger.critical("Maximum particle size must be a nonnegative float.")
                    print_usage()
                    error_encountered = True
            except Exception as e:
                logger.critical(f"Could not parse 'max_particle_size' argument: '{e}'")
                print_usage()
                max_particle_size = None
                error_encountered = True
            arguments_parsed["max_particle_size"] = max_particle_size
        elif name == "num_frames":
            # Set it to a really high number if max, I don't really wanna pass the number of frames into this function just to set this value accurately.
            if value == "max":
                value = 1e12 # 5e10 seconds ~ 1585 years
            try:
                number_of_frames = int(value)
                if number_of_frames < 0:
                    logger.critical("Number of frames must be a positive integer.")
                    print_usage()
                    error_encountered = True
            except Exception as e:
                logger.critical(f"Could not parse 'num_frames' argument: '{e}'")
                print_usage()
                number_of_frames = None
                error_encountered = True
            arguments_parsed["num_frames"] = number_of_frames
        elif name == "search_range":
            try:
                search_range = int(value)
                if search_range < 0:
                    logger.critical("Search range must be a nonnegative integer.")
                    print_usage()
                    error_encountered = True
            except Exception as e:
                logger.critical(f"Could not parse 'search_range' argument: '{e}'")
                print_usage()
                search_range = None
                error_encountered = True
            arguments_parsed["search_range"] = search_range
        elif name == "memory":
            try:
                memory = int(value)
                if memory < 0:
                    logger.critical("Memory must be a nonnegative integer.")
                    print_usage()
                    error_encountered = True
            except Exception as e:
                logger.critical(f"Could not parse 'memory' argument: '{e}'")
                print_usage()
                memory = None
                error_encountered = True
            arguments_parsed["memory"] = memory
        elif name == "min_traj_len":
            try:
                min_traj_len = int(value)
                if min_traj_len < 0:
                    logger.critical("Minimum trajectory length must be a nonnegative integer.")
                    print_usage()
                    error_encountered = True
            except Exception as e:
                logger.critical(f"Could not parse 'min_traj_len' argument: '{e}'")
                print_usage()
                min_traj_len = None
                error_encountered = True
            arguments_parsed["min_traj_len"] = min_traj_len
        elif name == "force_regenerate":
            try:
                force_regenerate_bool = int(value)
                if force_regenerate_bool == 0:
                    force_regenerate_bool = False
                else:
                    force_regenerate_bool = True
            except Exception as e:
                logger.critical(f"Could not parse 'force_regenerate' argument: '{e}'")
                print_usage()
                force_regenerate_bool = None
                error_encountered = True
            arguments_parsed["force_regenerate"] = force_regenerate_bool
    
    # Needs to be checked first, since 'num_frames' parameter may depend on it.
    if not ("min_traj_len" in arguments_parsed):
        logger.warning(f"Parameter 'min_traj_len' is missing, defaulting to '{DEFAULT_MIN_TRAJ_LEN}'.")
        arguments_parsed["min_traj_len"] = DEFAULT_MIN_TRAJ_LEN
    
    if not ("tune" in arguments_parsed):
        logger.warning(f"Parameter 'tune' is missing, defaulting to '{DEFAULT_TUNE_VALUE}'.")
        arguments_parsed["tune"] = DEFAULT_TUNE_VALUE
    if not ("particle_diameter" in arguments_parsed):
        logger.warning(f"Parameter 'particle_diameter' is missing, defaulting to '{DEFAULT_PARTICLE_DIAMETER}'.")
        arguments_parsed["particle_diameter"] = DEFAULT_PARTICLE_DIAMETER
    if not ("minmass" in arguments_parsed):
        logger.warning(f"Parameter 'minmass' is missing, defaulting to '{DEFAULT_MINMASS}'.")
        arguments_parsed["minmass"] = DEFAULT_MINMASS
    if not ("max_particle_size" in arguments_parsed):
        logger.warning(f"Parameter 'max_particle_size' is missing, defaulting to '{DEFAULT_MAX_PARTICLE_SIZE}'.")
        arguments_parsed["max_particle_size"] = DEFAULT_MAX_PARTICLE_SIZE
    if not ("num_frames" in arguments_parsed):
        logger.warning(f"Parameter 'num_frames' is missing, defaulting to '{DEFAULT_MIN_TRAJ_LEN}'.")
        arguments_parsed["num_frames"] = DEFAULT_MIN_TRAJ_LEN
    else:
        if arguments_parsed["tune"]:
            logger.warning(f"Setting 'num_frames' to '{arguments_parsed['min_traj_len']}', as this is most efficient when tuning.")
            arguments_parsed["num_frames"] = arguments_parsed["min_traj_len"]
    if not ("search_range" in arguments_parsed):
        logger.warning(f"Parameter 'search_range' is missing, defaulting to '{DEFAULT_SEARCH_RANGE}'.")
        arguments_parsed["search_range"] = DEFAULT_SEARCH_RANGE
    if not ("memory" in arguments_parsed):
        logger.warning(f"Parameter 'memory' is missing, defaulting to '{DEFAULT_MEMORY}'.")
        arguments_parsed["memory"] = DEFAULT_MEMORY
    if not ("force_regenerate" in arguments_parsed):
        logger.warning(f"Parameter 'force_regenerate' is missing, defaulting to '{DEFAULT_FORCE_REGENERATE}'.")
        arguments_parsed["force_regenerate"] = DEFAULT_FORCE_REGENERATE
    
    # Check if the number of frames is less than the minimum trajectory length in frames in filter_stubs.
    if arguments_parsed["num_frames"] < arguments_parsed["min_traj_len"]:
        logger.warning(f"Parameter 'num_frames' is less than arguments_parsed[\"min_traj_len\"]={arguments_parsed['min_traj_len']}, defaulting to {arguments_parsed['min_traj_len']} (all trajectories will be purged otherwise, sorry that this is needed, the code setup sort of requires it).")
        arguments_parsed["num_frames"] = arguments_parsed["min_traj_len"]
    
    # Exit the program if necessary.
    if error_encountered:
        sys.exit(0)
    return arguments_parsed

def main(cmd_args: list[str]):
    parameters = {
        "500nm":  ["particle_diameter=15", "minmass=400", "max_particle_size=3"], 
        "750nm":  ["particle_diameter=15", "minmass=1000", "max_particle_size=3"], 
        "1000nm": ["particle_diameter=15", "minmass=1000", "max_particle_size=3.5"], 
        "1500nm": ["particle_diameter=15", "minmass=3000", "max_particle_size=5"], 
        "2000nm": ["particle_diameter=19", "minmass=3500", "max_particle_size=5.5"]
    }
    universal_parameters = ["tune=0", "force_regenerate=1", "num_frames=800"]
    
    for particle_size, params in parameters.items():
        for sample_id in ["A", "B", "C"]:
            for recording_id in [1, 2, 3]:
                # Set video path.
                #video_path = "data/metingen/005_mass_percent_500nm_40x_A_1.wmv"
                video_path = f"data/metingen/005_mass_percent_{particle_size}_40x_{sample_id}_{recording_id}.wmv"
                
                # Parse command line arguments. This function substitutes default values for missing parameters.
                cmd_args = params + universal_parameters
                args_parsed = parse_cmd_args(cmd_args)
                
                # Load data.
                frames, batch_data, link_data = load_data(video_path, args_parsed)
                
                # Run tuning function if tune=True.
                if args_parsed["tune"]:
                    tune_parameters.show_tune(frames, os.path.basename(video_path), particle_diameter=args_parsed["particle_diameter"], minimum_mass=args_parsed["minmass"])
                else:
                    # Check if batch_data contains too few frames, if so regenerate data.
                    number_of_frames_loaded = len(list(set(batch_data["frame"])))
                    if args_parsed["num_frames"] > number_of_frames_loaded:
                        logger.warning(f"Regenerating batch- and link data, since tune=True and number of frames currently loaded ({number_of_frames_loaded}) is less than number of frames requested ({args_parsed['num_frames']}).")
                        frames, batch_data, link_data = load_data(video_path, args_parsed, force_regenerate=True)
                    
                    # Construct output_data_dict to be used by the functions to store trajectory data and particle positions.
                    output_data_dict = {}
                    
                    # Add metadata to command line arguments (the ones that were missing have been set to some default value by the parse_cmd_args function).
                    metadata = {k: v for k, v in args_parsed.items()}
                    metadata["video_path"] = video_path
                    metadata["total_number_of_frames_in_video"] = len(frames)
                    output_data_dict["metadata"] = metadata
                    
                    # Generate trajectories.
                    trajectory_data = get_trajectories(frames, batch_data, link_data, args_parsed, output_data_dict)
                    
                    # Calculate diffusion constant.
                    diffusion_constant = analyse_trajectories(trajectory_data, args_parsed, output_data_dict)
                    
                    # Save output data dict to file.
                    output_data_folder = os.path.join(os.getcwd(), "gen")
                    if not os.path.isdir(output_data_folder):
                        os.mkdir(output_data_folder)
                    output_file_path = os.path.join(output_data_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_output_data.pickle")
                    with open(output_file_path, "wb") as file:
                        pickle.dump(output_data_dict, file)
                    logger.info(f"Saved output data dict to '{output_file_path}'.")

if __name__ == "__main__":
    main(cmd_args=sys.argv[1:])
