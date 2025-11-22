import os
import glob
import json
import math
import re
from pyhdf.SD import SD, SDC

# MODIS Sinusoidal Projection Parameters
R = 6371007.181  # Radius of the Earth in meters

def sinusoidal_to_latlon(x, y):
    """
    Convert Sinusoidal projection coordinates (x, y) in meters to Latitude/Longitude in degrees.
    """
    phi = y / R
    # Avoid division by zero at poles
    if math.isclose(abs(phi), math.pi / 2, rel_tol=1e-9):
        lam = 0
    else:
        lam = x / (R * math.cos(phi))
    
    lat = math.degrees(phi)
    lon = math.degrees(lam)
    
    return lat, lon

def get_tile_boundaries(hdf_path):
    """
    Extracts the boundary coordinates from a MODIS HDF file.
    """
    try:
        hdf = SD(hdf_path, SDC.READ)
        # StructMetadata.0 contains the projection coordinates
        struct_metadata = hdf.attributes().get("StructMetadata.0")
        hdf.end()
        
        if not struct_metadata:
            print(f"Warning: No StructMetadata.0 found in {hdf_path}")
            return None

        # Parse UpperLeftPointMtrs and LowerRightMtrs
        # Format example: UpperLeftPointMtrs=(-5559752.598333,5559752.598333)
        ul_match = re.search(r'UpperLeftPointMtrs=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', struct_metadata)
        lr_match = re.search(r'LowerRightMtrs=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', struct_metadata)
        
        if not ul_match or not lr_match:
            print(f"Warning: Could not parse coordinates in {hdf_path}")
            return None
            
        ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
        lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
        
        # Calculate 4 corners
        # UL: (ul_x, ul_y)
        # UR: (lr_x, ul_y)
        # LL: (ul_x, lr_y)
        # LR: (lr_x, lr_y)
        
        corners = {
            "UL": sinusoidal_to_latlon(ul_x, ul_y),
            "UR": sinusoidal_to_latlon(lr_x, ul_y),
            "LL": sinusoidal_to_latlon(ul_x, lr_y),
            "LR": sinusoidal_to_latlon(lr_x, lr_y)
        }
        
        return corners

    except Exception as e:
        print(f"Error processing {hdf_path}: {e}")
        return None

def main():
    base_dir = "submission_data"
    output_file = "tile_latlon.json"
    
    tile_data = {}
    
    # Iterate through tile directories
    tile_dirs = glob.glob(os.path.join(base_dir, "h*v*"))
    
    print(f"Found {len(tile_dirs)} tile directories.")
    
    for tile_dir in tile_dirs:
        tile_name = os.path.basename(tile_dir)
        
        # Find the first HDF file
        hdf_files = glob.glob(os.path.join(tile_dir, "*.hdf"))
        if not hdf_files:
            print(f"No HDF files found in {tile_dir}")
            continue
            
        # Use the first file to get boundaries (assuming all files in the tile have same bounds)
        first_hdf = hdf_files[0]
        print(f"Processing {tile_name} using {os.path.basename(first_hdf)}...")
        
        boundaries = get_tile_boundaries(first_hdf)
        
        if boundaries:
            tile_data[tile_name] = {
                "boundaries": boundaries,
                "center": {
                    "lat": (boundaries["UL"][0] + boundaries["LR"][0]) / 2,
                    "lon": (boundaries["UL"][1] + boundaries["LR"][1]) / 2
                }
            }
            
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(tile_data, f, indent=2)
        
    print(f"Saved tile boundaries to {output_file}")

if __name__ == "__main__":
    main()
