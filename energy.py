import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import torch
import shutil
import os
import cv2
import re
import logging
from tqdm import tqdm

LOG = logging.getLogger('tube_optimizer')

def plot_tubes(ax, tube_data, tube_starts, areas, title):
    colors = plt.cm.viridis(np.linspace(0, 1, len(tube_data)))
    for i, (tube_id, points) in enumerate(tube_data.items()):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if tube_starts is not None:
            z = z - np.min(z) + tube_starts[i]
        area_size = (areas[tube_id] / np.max(list(areas.values())) * 100)
        ax.scatter(x, y, z, s=area_size, c=[colors[i % len(colors)]], label=f'Tube {tube_id}')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Time (t)')


def parse_contours(contour_str):
    """
    Parse the contour string from CSV to a numpy array suitable for OpenCV processing.
    """
    # Remove unwanted characters and clean up the format
    clean_str = re.sub(r'\[\[|\]\]', '', contour_str)  # Remove double brackets
    clean_str = re.sub(r'\s+', ' ', clean_str)  # Reduce multiple spaces to single spaces
    clean_str = clean_str.replace(' ', ',')  # Replace remaining spaces with commas for consistent splitting
    clean_str = clean_str.strip()  # Remove leading/trailing whitespace

    # Split into pairs and parse as integers
    pairs = clean_str.split('],[')
    points = []
    for pair in pairs:
        try:
            point = tuple(map(int, pair.split(',')))
            points.append(point)
        except ValueError:
            continue  # Skip any malformed pairs

    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)

def plot_contours(contours):
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=2)
        hull = cv2.convexHull(contour)
        plt.plot(hull[:, 0, 0], hull[:, 0, 1], 'g', linewidth=2)
    plt.show()

def parse_contours(contour_str):
    """
    Parse the contour string from CSV to a numpy array suitable for OpenCV processing.
    Assumes contours are given in the format like: '[[[310 35]] [[311 36]] ...]'
    """
    # Extract numbers using a regex pattern that finds pairs of numbers
    points = re.findall(r'\[\[\[(\d+)\s+(\d+)\]\]\]', contour_str)
    
    # Convert list of string tuples into integers
    array_points = [tuple(map(int, point)) for point in points]

    # Convert list of tuples to a NumPy array with shape (-1, 1, 2)
    return np.array(array_points, dtype=np.int32).reshape(-1, 1, 2)


def compute_convex_hull(contours):
    """
    Compute and return the convex hull for the given set of points.
    """
    if contours.shape[0] < 3:  # Not enough points to compute a convex hull
        return contours
    return cv2.convexHull(contours)

def load_tubes(files_pattern):
    tube_data = {}
    areas = {}
    boxes = {}
    original_tube_map = {}
    full_dataframes = {}
    tube_id_counter = 0
    
    for file_path in glob.glob(files_pattern):
        file_name = os.path.basename(file_path)
        original_tube_id = int(file_name.split('node')[0]) 
        
        df = pd.read_csv(file_path)
        df['centroid_x'] = (df['x1'] + df['x2']) / 2
        df['centroid_y'] = (df['y1'] + df['y2']) / 2
        df['area'] = abs((df['x2'] - df['x1']) * (df['y2'] - df['y1']))
        df_sorted = df.sort_values(by='time')
        time_diffs = df_sorted['time'].diff()
        split_points = time_diffs > 1

        for is_new_tube, group in df_sorted.groupby(split_points.cumsum()):
            tube_data[tube_id_counter] = group[['centroid_x', 'centroid_y', 'time']].values
            areas[tube_id_counter] = group['area'].mean()
            boxes[tube_id_counter] = (group['x1'].min(), group['y1'].min(), group['x2'].max(), group['y2'].max())
            original_tube_map[tube_id_counter] = original_tube_id
            full_dataframes[tube_id_counter] = group
            tube_id_counter += 1

    return tube_data, areas, boxes, original_tube_map, full_dataframes


def get_time_bounds(tube, start_time):
    """Calculate the start and end time of a tube."""
    duration = torch.tensor(np.max(tube[:, 2]) - np.min(tube[:, 2]))
    return start_time, start_time + duration


def compute_iou(box_i, box_j):
    # Convert lists/tuples to tensors for easier manipulation
    box_i = box_i.clone().detach().requires_grad_(True)
    box_j = box_j.clone().detach().requires_grad_(True)

    # Calculate intersection coordinates
    xA = torch.max(box_i[0], box_j[0])
    yA = torch.max(box_i[1], box_j[1])
    xB = torch.min(box_i[2], box_j[2])
    yB = torch.min(box_i[3], box_j[3])

    # Compute the area of intersection
    interArea = torch.max(torch.tensor(0.0), xB - xA) * torch.max(torch.tensor(0.0), yB - yA)

    # Compute the area of both bounding boxes
    box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
    box_j_area = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

    # Compute the union area
    unionArea = box_i_area + box_j_area - interArea

    # Compute the IoU
    iou = interArea / unionArea if unionArea > 0 else torch.tensor(0.0)
    return iou


def compute_energy(tube_data:dict, 
                   starts:float, 
                   areas:dict, 
                   boxes:dict, 
                   total_frames:int,
                   device:str):
    collision_energy = torch.tensor(0.0, dtype=torch.float32, device=device)
    chronological_energy = torch.tensor(0.0, dtype=torch.float32, device=device)
    activity_energy = torch.tensor(0.0, dtype=torch.float32, device=device)
    temporal_energy = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    # Adjust based on your specific implementation needs
    video_start, video_end = torch.tensor(0, dtype=torch.float32, device=device), torch.tensor(total_frames, dtype=torch.float32, device=device)

    for i, tube_id_i in enumerate(tube_data):
        start_i = starts[i].float().to(device)
        end_i = start_i + torch.tensor(tube_data[tube_id_i][:, 2].ptp(), dtype=torch.float32, device=device)
        
        # Convert boxes to tensors
        box_i = torch.tensor(boxes[tube_id_i], dtype=torch.float32, device=device)

        # Activity loss: penalizing activities out of video temporal boundary
        if start_i < video_start or end_i > video_end:
            activity_energy += abs(start_i - video_start) + abs(end_i - video_end)

        for j, tube_id_j in enumerate(tube_data):
            if i == j:
                continue
            start_j = starts[j].float().to(device)
            end_j = start_j + torch.tensor(tube_data[tube_id_j][:, 2].ptp(), dtype=torch.float32, device=device)
            
            # Convert boxes to tensors
            box_j = torch.tensor(boxes[tube_id_j], dtype=torch.float32, device=device)

            # Compute temporal overlap
            overlap_time = torch.max(torch.tensor(0.0, dtype=torch.float32, device=device), torch.min(end_i, end_j) - torch.max(start_i, start_j))
            if overlap_time > 0:
                iou = compute_iou(box_i, box_j)  # Assuming `compute_iou` handles tensors
                collision_energy += iou * overlap_time

            # Chronological loss: penalizing chronological errors
            if (start_i > start_j and end_i < end_j) or (start_i < start_j and end_i > end_j):
                chronological_energy += abs(start_i - start_j)

        # Calculate temporal energy based on the gaps
        next_start = starts[i + 1].float().to(device) if i + 1 < len(starts) else end_i
        temporal_gap = torch.clamp(next_start - end_i, min=0)
        temporal_energy += temporal_gap ** 2

    total_energy = collision_energy + chronological_energy + activity_energy + temporal_energy
    return total_energy, collision_energy, chronological_energy, activity_energy, temporal_energy


def optimize_tube_starts(tube_data:dict, areas:dict, boxes:dict, total_frames:int=1474, epochs:int=1000, device:str='cpu'):
    # Setup logging for this function
    if not LOG.hasHandlers():  # Ensure no duplicate handlers are added
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
        LOG.setLevel(logging.INFO)

    tube_ids = list(tube_data.keys())
    starts = torch.nn.Parameter(torch.zeros(len(tube_ids), dtype=torch.float, device=device))  # Start with zeros on device
    optimizer = torch.optim.Adam([starts], lr=0.01)  # Using Adam for better performance

    for epoch in tqdm(range(epochs), desc='Optimization'):  
        optimizer.zero_grad()
        total_energy, collision_energy, chronological_energy, activity_energy, temporal_energy = compute_energy(tube_data, starts, areas, boxes, total_frames, device)
        total_energy.backward()
        optimizer.step()

        # Clamping starts to non-negative values and ensuring sequential order
        with torch.no_grad():
            starts.sort()
            starts.clamp_(min=0)

        if epoch % 500 == 0:
            LOG.info(f"Epoch {epoch}: Total Energy = {total_energy.item()}, Collision = {collision_energy.item()}, Temporal = {temporal_energy.item()}")

    return {tube_id: start.item() for tube_id, start in zip(tube_ids, starts)}


def plot_specific_tubes(ax, tube_data:dict, tube_starts:float, tube_ids:int, areas:dict, title:str):
    colors = plt.cm.viridis(np.linspace(0, 1, len(tube_ids)))
    
    for i, tube_id in enumerate(tube_ids):
        points = tube_data[tube_id]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if tube_starts is not None:
            z = z - np.min(z) + tube_starts[i]
        area_size = (areas[tube_id] / np.max(list(areas.values())) * 100)
        ax.scatter(x, y, z, s=area_size, color=colors[i % len(colors)], label=f'Tube {tube_id}')
    
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Time (t)')
    ax.legend()
    
def save_optimized_tubes(full_dataframes:dict, 
                         optimized_starts:float, 
                         original_tube_map:dict, 
                         output_dir:str):
    for tube_id, start_time in zip(full_dataframes.keys(), optimized_starts):
        df = full_dataframes[tube_id]
        original_id = original_tube_map[tube_id]
        
        # Calculate the offset for curr_time based on optimized start time
        time_offset = start_time - df['time'].min()
        df['time'] = df['time'] + time_offset
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"optimized_tube_{original_id}.csv")
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, mode='w', header=True, index=False)
        
        # print(f"Optimized data for Tube {original_id} saved to {output_path}.")

def convert_directory_csv_to_txt(input_dir:str, 
                                 output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(input_dir, filename)
            txt_filename = filename.replace('.csv', '.txt')
            txt_file_path = os.path.join(output_dir, txt_filename)
            csv_to_txt(csv_file_path, txt_file_path)



def csv_to_txt(csv_file_path:str, txt_file_path:str):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Assume 'T' corresponds to the original tube ID extracted from the filename
    # This requires that the tube ID be part of the filename or you pass it somehow.
    tube_id = csv_file_path.split('/')[-1].split('.')[0].split('_')[-1].zfill(4)
    
    # Extract and format the required columns
    # We need to ensure the DataFrame contains these columns; this assumes they exist.
    df['T'] = tube_id  # Set all rows to have this tube ID
    df['time'] = df['time']  # Rename or ensure this is the right column to use
    
    # Select the specific columns and format the output
    output_columns = ['T', 'n', 'x1', 'x2', 'y1', 'y2', 'time']
    if 'n' not in df.columns:
        df['n'] = range(1, len(df) + 1)  # Assuming 'n' should be a sequence number
    
    # Format each row as a string with the desired output format
    formatted_data = df[output_columns].apply(
        lambda x: f"{x['T']}, {x['n']}, {x['x1']}, {x['x2']}, {x['y1']}, {x['y2']}, {x['time']:.2f},",
        axis=1
    )
    
    # Write to a text file
    with open(txt_file_path, 'w') as f:
        for line in formatted_data:
            f.write(line + '\n')

    # print(f"Data from {csv_file_path} has been converted and saved to {txt_file_path}.")

def optimize_tube(files_pattern:str="*/*.csv", output_dir:str='../optimized_tubes', video_length:int=1474, epochs:int=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'⏳[Info] Device for optimization :**{device}** \u2705')
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    tube_data, areas, boxes, original_tube_map, full_dataframes = load_tubes(files_pattern)
    optimized_starts = optimize_tube_starts(tube_data, areas, boxes, video_length, epochs,device=device)
    save_optimized_tubes(full_dataframes, optimized_starts, original_tube_map, output_dir)
    convert_directory_csv_to_txt(output_dir, output_dir)
    
    # Plot initial and optimized arrangements
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_tubes(ax1, tube_data, None, areas, 'Initial Arrangement')
    ax2 = fig.add_subplot(122, projection='3d')
    plot_tubes(ax2, tube_data, optimized_starts, areas, 'Optimized Arrangement')
    
    plt.tight_layout()
    
    # Save the plot to the ./ directory (or another desired location)
    output_plot_path = './optimized_tubes_plot.png'
    plt.savefig(output_plot_path)
    plt.close()