# FOREST COVER VISUALIZATIONS
# Run this code in your notebook to create heatmaps and visualizations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Set up plotting style
plt.style.use('default')
sns.set_palette("viridis")

def create_forest_cover_heatmap(data, year, title_suffix="", figsize=(12, 10)):
    """
    Create a heatmap visualization of forest cover for a given year.
    
    Args:
        data: 2D numpy array of forest cover percentages
        year: year for the title
        title_suffix: additional text for the title
        figsize: figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap for forest cover (green tones)
    colors = ['#8B4513', '#DEB887', '#90EE90', '#228B22', '#006400']  # Brown to dark green
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('forest', colors, N=n_bins)
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Forest Cover (%)', fontsize=12)
    
    # Set title and labels
    ax.set_title(f'Forest Cover Heatmap - {year}{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel Index (X)', fontsize=12)
    ax.set_ylabel('Pixel Index (Y)', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        stats_text = f'Mean: {np.mean(valid_data):.1f}%\nStd: {np.std(valid_data):.1f}%\nMin: {np.min(valid_data):.1f}%\nMax: {np.max(valid_data):.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def create_time_series_comparison(data, years, pixel_coords, figsize=(15, 8)):
    """
    Create time series comparison for specific pixels.
    
    Args:
        data: 3D numpy array (time_steps, height, width)
        years: list of years
        pixel_coords: list of (i, j) tuples for pixels to plot
        figsize: figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(pixel_coords)))
    
    for idx, (i, j) in enumerate(pixel_coords):
        time_series = data[:, i, j]
        valid_mask = ~np.isnan(time_series)
        
        if np.sum(valid_mask) > 0:
            ax.plot(np.array(years)[valid_mask], time_series[valid_mask], 
                   marker='o', linewidth=2, markersize=6, 
                   color=colors[idx], label=f'Pixel ({i}, {j})')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Forest Cover (%)', fontsize=12)
    ax.set_title('Forest Cover Time Series for Selected Pixels', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def create_change_map(data_early, data_late, year_early, year_late, figsize=(15, 6)):
    """
    Create a map showing forest cover change between two years.
    
    Args:
        data_early: 2D array for early year
        data_late: 2D array for late year
        year_early: early year
        year_late: late year
        figsize: figure size tuple
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Early year
    im1 = ax1.imshow(data_early, cmap='Greens', vmin=0, vmax=100)
    ax1.set_title(f'Forest Cover - {year_early}', fontsize=12)
    ax1.set_xlabel('Pixel Index (X)')
    ax1.set_ylabel('Pixel Index (Y)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Late year
    im2 = ax2.imshow(data_late, cmap='Greens', vmin=0, vmax=100)
    ax2.set_title(f'Forest Cover - {year_late}', fontsize=12)
    ax2.set_xlabel('Pixel Index (X)')
    ax2.set_ylabel('Pixel Index (Y)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Change map
    change = data_late - data_early
    im3 = ax3.imshow(change, cmap='RdBu_r', vmin=-50, vmax=50)
    ax3.set_title(f'Change ({year_late} - {year_early})', fontsize=12)
    ax3.set_xlabel('Pixel Index (X)')
    ax3.set_ylabel('Pixel Index (Y)')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Change (%)')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)

def create_forecast_visualization(historical_data, forecast_data, historical_years, forecast_years, 
                                 pixel_coords, figsize=(15, 6)):
    """
    Create visualization showing historical data and forecasts.
    
    Args:
        historical_data: 3D array of historical data
        forecast_data: 3D array of forecast data
        historical_years: list of historical years
        forecast_years: list of forecast years
        pixel_coords: list of (i, j) tuples for pixels to plot
        figsize: figure size tuple
    """
    fig, (ax1) = plt.subplots(1, 1, figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(pixel_coords)))
    
    # Plot 1: Time series for selected pixels
    for idx, (i, j) in enumerate(pixel_coords):
        # Historical data
        hist_ts = historical_data[:, i, j]
        valid_mask = ~np.isnan(hist_ts)
        
        if np.sum(valid_mask) > 0:
            ax1.plot(np.array(historical_years)[valid_mask], hist_ts[valid_mask], 
                    marker='o', linewidth=2, markersize=6, 
                    color=colors[idx], label=f'Pixel ({i}, {j}) - Historical')
        
        # Forecast data
        forecast_ts = forecast_data[:, i, j]
        valid_forecast = ~np.isnan(forecast_ts)
        
        if np.sum(valid_forecast) > 0:
            ax1.plot(np.array(forecast_years)[valid_forecast], forecast_ts[valid_forecast], 
                    marker='s', linewidth=2, markersize=6, linestyle=':',
                    color=colors[idx], alpha=0.7, label=f'Pixel ({i}, {j}) - Forecast')
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Forest Cover (%)', fontsize=12)
    ax1.set_title('Historical Data and Forecasts for Selected Pixels', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of forecast vs historical average
    # Compute means robustly to avoid RuntimeWarning when all-NaN along axis
    hist_count = np.sum(~np.isnan(historical_data), axis=0)
    hist_sum = np.nansum(historical_data, axis=0, dtype=np.float64)
    historical_mean = np.divide(
        hist_sum,
        hist_count,
        out=np.full(hist_sum.shape, np.nan),
        where=hist_count > 0,
    )

    fc_count = np.sum(~np.isnan(forecast_data), axis=0)
    fc_sum = np.nansum(forecast_data, axis=0, dtype=np.float64)
    forecast_mean = np.divide(
        fc_sum,
        fc_count,
        out=np.full(fc_sum.shape, np.nan),
        where=fc_count > 0,
    )
    
    plt.tight_layout()
    return fig, (ax1)

# Example usage code (uncomment to run):
"""
# Load your data first (assuming you have the variables from previous cells)
# all_time_series = load_time_series_data(sorted_paths, 'Percent_Tree_Cover')
# years = [r['acquisition_dt'].year for r in records_sorted]

# Create visualizations
print("Creating forest cover visualizations...")

# 1. Heatmap for first year
fig1, ax1 = create_forest_cover_heatmap(all_time_series[0], years[0])
plt.show()

# 2. Heatmap for last year
fig2, ax2 = create_forest_cover_heatmap(all_time_series[-1], years[-1])
plt.show()

# 3. Time series for selected pixels
pixel_coords = [(100, 100), (200, 200), (300, 300)]
fig3, ax3 = create_time_series_comparison(all_time_series, years, pixel_coords)
plt.show()

# 4. Change map between first and last year
fig4, axes = create_change_map(all_time_series[0], all_time_series[-1], years[0], years[-1])
plt.show()

print("Visualizations completed!")
"""

print("Forest cover visualization functions loaded!")
print("Use these functions to create heatmaps and time series plots.")
