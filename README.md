# cse6242-Team-Iotas
Code for Team Iotas to visualize past forest loss globally and explore machine learning models to predict future deforestation.


Steps:
1. Download this repo, set up environment by importing or 'cond env create -f forest-viewer.yml'
2. Download Data from https://search.earthdata.nasa.gov/ into data folder as hdf files. Remember to use \*hXXv.XX\* formatting, adding stars to the front and end to get the whole timerange for the selected tile as you pick it. (you can skip this step for demo, as our repo already has the processed data in data/processed_tiles_with_forecast.json)
3. Run get_tile_latlon.py to obtain each tile's (latitude, longitude) information from hdf files (you can also skip this step for demo, as our repo already has tile_latlon.json)
4. From the main project folder, run 'python -m http.server 8000'
5. If it has not opened go to 'http://localhost:8000/visualize.html'
