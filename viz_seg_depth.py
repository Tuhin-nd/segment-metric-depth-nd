import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get the current working directory
current_working_directory = os.getcwd()

# Define the paths to the folders
# segmentation_folder = os.path.join(current_working_directory, 'segmentation_88641cb95a')
# depth_folder = os.path.join(current_working_directory, 'metric_depth_88641cb95a')
# output_folder = os.path.join(current_working_directory, 'output_overlay_images')

segmentation_folder = os.path.join(current_working_directory, 'segmentation_yvis_2021_tennis')
depth_folder = os.path.join(current_working_directory, 'output_metric_depth_fb104c286f')
output_folder = os.path.join(current_working_directory, 'output_overlay_images_fb104c286f')

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
# List all files in the segmentation folder
segmentation_files = sorted(os.listdir(segmentation_folder))

# Process each file
for seg_file in segmentation_files:
    # Skip hidden files
    if seg_file.startswith('.'):
        continue
    # Construct the full path to the segmentation image and depth data
    segmentation_image_path = os.path.join(segmentation_folder, seg_file)
    depth_data_path = os.path.join(depth_folder, seg_file.replace('.jpg', '.npy'))

    # Load the segmentation image
    segmentation_image = cv2.imread(segmentation_image_path)

    # Load the depth data
    depth_data = np.load(depth_data_path)

    # Ensure both images are of the same size
    height, width = segmentation_image.shape[:2]
    depth_data = cv2.resize(depth_data, (width, height))

    # Overlay the segmentation image on the depth data
    overlay_image = cv2.addWeighted(segmentation_image, 0.5, cv2.cvtColor(depth_data.astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.5, 0)

    # Calculate average depth values for every 75x75 box
    box_size = 75
    avg_depth_values = []
    box_number = 0
    for y in range(0, height, box_size):
        for x in range(0, width, box_size):
            box = depth_data[y:y+box_size, x:x+box_size]
            avg_depth = np.mean(box)
            avg_depth_values.append((box_number, x, y, avg_depth))
            box_number += 1

    # Display the results
    for (box_number, x, y, avg_depth) in avg_depth_values:
        cv2.putText(overlay_image, f'{avg_depth:.2f}m', (x, y + box_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Save the overlaid image with average depth values
    output_image_path = os.path.join(output_folder, seg_file)
    cv2.imwrite(output_image_path, overlay_image)
