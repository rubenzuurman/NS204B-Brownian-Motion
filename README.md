# NS204B-Brownian-Motion

## Start
You can see if running '\_\_main\_\_.py' works. This will analyse all videos inside the 'data/metingen/' directory.<br />
The '\_\_main\_\_.py' file is kind of a mess, since I edited it pretty heavily to analyse all the videos in one go. Also the function for analysing command line arguments would ideally be in another file.<br />
In any case all dependencies can be found inside requirements.txt.<br />

The log folder contains logs for all the things that were done.<br />
The src folder contains all python files (except for \_\_main\_\_.py, I have this file because I can run using 'python -u .' and import using the folder structure, which I think is intuitive (example: from src.data\_loader import load_video)).<br />
&nbsp;&nbsp;&nbsp;&nbsp;cache.py contains functions for the caching behaviour<br />
&nbsp;&nbsp;&nbsp;&nbsp;data\_loader.py contains functionality for loading data and/or cache<br />
&nbsp;&nbsp;&nbsp;&nbsp;tune\_parameters.py contains code for generating an image for tuning the parameters (like particle diameter and minimum mass). This comes in handy if you want to quickly converge on usable parameters. (It's not really usable since the \_\_main\_\_.py file is messed up.)<br />
&nbsp;&nbsp;&nbsp;&nbsp;analysis.py contains the analysis code, currently only get\_trajectories() and analyse\_trajectories() (and subpx\_bias()) are used I think, the other functions are for loading, which I moved to the load_data.py file.<br />
The data folder contains the recordings. The 'official' experiment recordings are inside the folder 'data/metingen/'. The other videos are test videos from the start of the project.<br />
The gen folder is automatically generated when the project is run. It contains pickle files, which contain the particle position data for a specific video file, together with useful metadata. These pickle files are used to do the diffusion coefficient calculations and make the final plots.<br />
The cache folder is automatically generated when the project is run, and can be deleted safely. If no cache is present, the cache will be generated. Cache can come in handy when analysing the same video multiple times using the same parameters but a different plot configuration for example.<br />
Plenty of plots will be generated automatically for every analysed video, but they will overwrite the last. Add a 'break' statement to the end of the \_\_main\_\_.py for loop to only analyse one video (the first one, aka 500 nm A1, aka the stationary video).<br />

ScriptBRWN.py is a standalone file and depends on the pickle files. It generates the important plots for the report from the pickle files.
