# cse6242-Team-Iotas
Code for Team Iotas to visualize past forest loss globally and explore machine learning models to predict future deforestation.


Steps:
1. **Repo preparation**
    - Download this repo, set up environment by importing or 'cond env create -f forest-viewer.yml'
2. **Data preparation** (you can skip this step for demo, as our repo already has the processed data and tile coordinate data in data/processed_tiles_with_forecast.json and tile_latlon.json)
    - (optional for demo) Download MODIS Percent Tree Cover Data from https://search.earthdata.nasa.gov/ into data folder as hdf files. Remember to use \*hXXv.XX\* formatting, adding stars to the front and end to get the whole timerange for the selected tile as you pick it.
    - (optional for demo) Run hdf4_reader.py `python hdf4_reader.py` to convert hdf4 tree cover data into csv format.
    - (optional for demo) Run create_json.py `python create_json.py` to convert the csv data in the previous step to json format. This step augments the historical data with prediction data using our Adaptive Auto Regressive Model.
    - (optional for demo) Run get_tile_latlon.py `python get_tile_latlon.py` to obtain each tile's (latitude, longitude) information from the downloaded MODIS hdf4 files.
3. From the main project folder, run 'python -m http.server 8000'
4. If it has not opened go to 'http://localhost:8000/visualize.html'
