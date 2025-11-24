# cse6242-Team-Iotas
Team Iotas presents a visualization and forecasting tool for global forest cover.
The system allows users to:
- Shows past forest loss data in a selected area to the user
- Forecasts future forest loss in the selected area
- Successfully distinguishes between past and forecasted future data in a clear manner

We process MODIS Percent Tree Cover HDF4 files, convert them into rasters and apply AutoRegressive models to generate per pixel forest cover predictions. The final output is a browser based visualization that displays historical and predicted forest cover using a tile system.

The repo contains all components needed to preprocess the data, generate forecasts, and interactively visualize the results. 


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

a. Download the MODIS Percent Tree Cover Data from https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod44b-061
 into the data folder as hdf files.

Remember to use the h[HH]v[VV] regex to target a given location, adding stars to the front and end to get the whole timerange for the selected tile as you pick it.

Save HDF files into the data folder with subfolders created for each MODIS tile (e.g., data/h17v07). Each subfolder should contain all HDF files for that tile across the full time range.


b. Run get_tile_latlon.py python get_tile_latlon.py to obtain each tile's (latitude, longitude) information from the downloaded MODIS hdf4 files.

3. **Execution**
To launch the visualization:
   From the main project folder, run 'python -m http.server 8000'
   Open visualization in your browser by going to 'http://localhost:8000/visualize.html'
