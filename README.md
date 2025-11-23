# cse6242-Team-Iotas
Code for Team Iotas to visualize past forest loss globally and explore machine learning models to predict future deforestation.


Steps:
1. Download this repo, set up environment by importing or 'cond env create -f forest-viewer.yml'
2. Download Data from https://search.earthdata.nasa.gov/ into data folder as hdf files. Remember to use \*hXXv.XX\* formatting, adding stars to the front and end to get the whole timerange for the selected tile as you pick it.
3. Run get_tile_latlon.py to convert from hdf to json format
4. From the main project folder, run 'python -m http.server 8000'
5. If it has not opened go to 'http://localhost:8000/visualize.html'
