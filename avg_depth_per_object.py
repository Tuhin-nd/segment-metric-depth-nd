import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get the current working directory
current_working_directory = os.getcwd()

# Define the paths to the folders
depth_folder = os.path.join(current_working_directory, 'output_metric_depth_fb104c286f')
masks_folder = os.path.join(current_working_directory, 'output_with_masks_fb104c286f')

# List all depth files
depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.npy')])

# Dictionary to store average depth values for each object across frames
avg_depth_values = {}

# Process each depth file
for depth_file in depth_files:
    # Load the depth data
    depth_data = np.load(os.path.join(depth_folder, depth_file))

    # Get the base filename without extension
    base_filename = os.path.splitext(depth_file)[0]

    # Find corresponding mask files
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.startswith(base_filename) and f.endswith('.png')])

    # Process each mask file
    for mask_file in mask_files:
        # Load the mask image
        mask_image = cv2.imread(os.path.join(masks_folder, mask_file), cv2.IMREAD_GRAYSCALE)

        # Ensure the mask and depth data are of the same size
        mask_image = cv2.resize(mask_image, (depth_data.shape[1], depth_data.shape[0]))

        # Calculate the average depth for the masked region
        masked_depth_values = depth_data[mask_image > 0]
        if masked_depth_values.size == 0:
            avg_depth = np.NaN
        else:
            avg_depth = np.mean(masked_depth_values)


        # Extract object identifier from mask file name
        object_id = mask_file.split('_')[-1].split('.')[0]

        # Initialize the list for the object if not already present
        if object_id not in avg_depth_values:
            avg_depth_values[object_id] = []

        # Append the average depth value for the current frame
        avg_depth_values[object_id].append(avg_depth)

# Plot the average depth values for each object
plt.figure(figsize=(12, 8))

# Define a list of colors, markers, and line styles for different objects
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 's', 'D', '^', 'v', '<', '>']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
# object_id_map = {0: "Person 1", 1: "Bike", 2: "Person 2"}
# object_id_map = {0: "Car 1", 1: "Car 2"}
object_id_map = {0: "Player 0", 2 : "Player 2"}

# Plot each object's depth series
for idx, (object_id, depth_series) in enumerate(avg_depth_values.items()):
    if int(object_id) not in object_id_map:
        continue  # Skip if object_id is not in object_id_map
    x_values = range(len(depth_series))
    object_name = object_id_map.get(int(object_id), f'{object_id}')
    plt.plot(x_values, depth_series, label=object_name, 
             color=colors[idx % len(colors)], 
             marker=markers[idx % len(markers)], 
             linestyle=line_styles[idx % len(line_styles)], 
             linewidth=2)

# Add grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add labels and title
plt.xlabel('Frame #', fontsize=14)
plt.ylabel('Average Depth Value (metres)', fontsize=14)
plt.title('Average Depth Value per Object over Frames', fontsize=16)

# Add legend
plt.legend(fontsize=12)

# Save the plot as an image
plt.savefig('average_depth_per_object_fb104c286f.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
