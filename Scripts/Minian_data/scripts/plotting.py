import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os

def plot_cells(file_name, csv_path, save_figure=True):
    """
    Plot x cells' activity across frames using subplots for each cell
    
    Input: 
    - .csv file made by convert_to_csv

    Output:
    - A figure with individual subplots for each cell
    - Automatically saves as "Figure_1.pdf" to the minian folder

    Parameters:
    - file_name: name of the .csv file to load, must be string
    - csv_path: path to the .csv file, must be string. 
        Automatically uses minian output {dpath}
    - save_figure: bool, whether to save the figure (default True)
    
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Automatically detect all cell columns
    cell_columns = [col for col in df.columns if col != 'frame']
    num_cells = len(cell_columns)
    
    print(f"Found {num_cells}")
    
    # Create subplots - one for each cell
    fig, axes = plt.subplots(num_cells, 1, figsize=(12, 3*num_cells), sharex=True)
    
    # If only one cell, axes won't be an array, so we need to handle that
    if num_cells == 1:
        axes = [axes]
    
    # Generate colors for each cell
    colors = plt.cm.Set1(np.linspace(0, 1, num_cells))  # Use colormap for variety
    
    # Plot each cell in its own subplot
    for i, (cell, color) in enumerate(zip(cell_columns, colors)):
        axes[i].plot(df['frame'], df[cell], color=color, linewidth=1)
        axes[i].set_ylabel(f'{cell.replace("_", " ").title()} Activity', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{cell.replace("_", " ").title()} Activity Over Time', fontsize=11)
    
    # Set x-label only for bottom plot
    axes[-1].set_xlabel('Frame Number', fontsize=12)
    fig.suptitle(f'Individual Cell Activities Across Frames ({num_cells} cells)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_figure:
        # Get the directory from csv_path
        output_dir = os.path.dirname(csv_path)
        save_path = os.path.join(output_dir, "Figure_1.pdf")
        # Save the figure
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()

# Example usage
if __name__ == "__main__":    
    # Check if our CSV file exists, add path to your .csv file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    minian_dir = os.path.join(script_dir, "minian")
    param_save_minian = {"dpath": minian_dir}
    file_name = "C_binary.csv"  # specify a file name, must be string
    csv_path = os.path.join(param_save_minian["dpath"], file_name)
    print(f"Looking for CSV file at: {csv_path}")
    
    if os.path.exists(csv_path):
        plot_cells(file_name=file_name, csv_path=csv_path)
    else:
        print("CSV file not found! Please check the path. "
              f"\n Current path: {param_save_minian['dpath']}")
