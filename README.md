# cse6242-Team-Iotas
Team Iotas presents a visualization and forecasting tool for global forest cover.
The system allows users to:
- Shows past forest loss data in a selected area to the user
- Forecasts future forest loss in the selected area
- Successfully distinguishes between past and forecasted future data in a clear manner

We process MODIS Percent Tree Cover HDF4 files, convert them into rasters and apply AutoRegressive models to generate per pixel forest cover predictions. The final output is a browser based visualization that displays historical and predicted forest cover using a tile system.

The repo contains all components needed to preprocess the data, generate forecasts, and interactively visualize the results. 

Installation Demo Video to Follow Along: https://youtu.be/5pEL_PER8eg

Installation 
1. **Environment Set Up**
    - Clone the repository
        - 'git clone <repo-url>'
        - 'cd cse6242-Team-Iotas'
       or 
    - Activate environment using the included YAML
        - 'cond env create -f forest-viewer.yml'
        - 'conda activate forest-viewer'
     
2. **Data Preparation**
(You may skip this section when running the demo, because processed data is included.)
The repo includes
    - data/processed_tiles_with_forecast.json (historical and predicted forest cover)
    - tile_latlon.json (geographic coordinates for each MODIS tile)
  
Optional: Full Data Processing Pipeline (to generate from scratch)

- Download the MODIS Percent Tree Cover Data from https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod44b-061 (enter e.g. '\*h19v03\*' in Granule ID(s) box to get the whole timerange for the selected tile as you pick it) into the data folder as hdf files (Save HDF files into the data folder with subfolders created for each MODIS tile (e.g., data/h17v07). Each subfolder should contain all HDF files for that tile across the full time range.).
- If instead you want to look at a given area of land rather than the opaquely named files, follow this: Go to the same url, clock Data Access, then on the row with EarthData Search click Download. Take the tour or skip out of it, but sign in if needed, it is free. Then go to the top left and select spatial, before clicking on the map to complete the shape surrounding the land you want to view. Then click temporal and then check the box for using a temporal data range, before clicking apply. When the matching collection box shows up, click the plus sign. If it doesn't, try creating a project by clicking the floppydisk icon in the top right, adding your own name. Either way, then click on 'My Project' in the top right, and click download. You will be prompted to open a window in which case do so, and you will have to sign in there too, after which you go back to the same button. After that, a new tab with loading files to download will appear, where you again click download files. Then a foldere with the files appears, select the files from that and move to the data folder in the git project locally.
- Run process_final.py `python process_final.py` (this will generate data/process_tile_with_forecast.json with future forecast with Auto Regression model too)
- Run get_tile_latlon.py `python get_tile_latlon.py` to obtain each tile's (latitude, longitude) information from the downloaded MODIS hdf4 files


3. **Execution**
To launch the visualization:
   From the main project folder, run 'python -m http.server 8000'
   Open visualization in your browser by going to 'http://localhost:8000/visualize.html'
