import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

# Function to check if a frame has a significant object
def has_significant_object(mask, threshold_area=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > threshold_area:
            return True
    return False


# def Tube():
#     """
#     Stitching Tubes to the background.
#     """
#     # MHS algorithm
#     # process_files(visualize=True, save_csv=True)
#     filenames = sorted(glob.glob("*/*.csv"))

#     for filename in tqdm(filenames):
#         df = pd.read_csv(filename)
#         for index, row in df.iterrows():
#                 try:
#                     Tube, n, x1, x2, y1, y2, curr_time = int(row['T']), int(row['n']), int(row['x1']), int(row['x2']), int(row['y1']), int(row['y2']), row['time']
#                     image_path = f"{str(Tube).zfill(4)}/{str(n).zfill(4)}{args['ext']}"
#                     mask_path = f"{str(final).zfill(4)}/{str(Tube).zfill(4)}/{str(n).zfill(4)}{args['ext']}"
#                     # Opening Object from Tube
#                     image = np.asarray(Image.open(image_path))
#                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                     # Opening Object from Mask
#                     mask = np.asarray(Image.open(mask_path))
#                     # print(image.shape, mask.shape)

#                     # Check the dimensions of the image and mask
#                     image_shape = image.shape[:2]
#                     mask_shape = mask.shape[:2]

#                     if image_shape != mask_shape:
#                         raise ValueError(f"ROI {image_shape} and mask {mask_shape} dimensions do not match \u274C")

#                     try:
#                         # Stitching object to the background
#                         background = medianOfFrame
#                         normal_clone = blend_roi_on_background(background=background, roi=image, roi_mask=mask, 
#                                                                x1=x1, x2=x2, y1=y1, y2=y2)
#                         background = background.copy()
#                         # print(normal_clone)
#                         cv2.imwrite(os.path.join(str(synopsis_frames),
#                                          f'{str(n).zfill(4)}' + args['ext']), normal_clone)
#                     except Exception as e:
#                         # Exception when ROI is out of bounds or size of the object needs to be changed
#                         print(f'[\u274C] Exception while stitching object to the background: {e}')
#                         raise e
#                     if int(Tube) > 0:
#                         try:
#                             # Write the constructed frame to the video
#                             cv2.startWindowThread()
#                             video.write(normal_clone)

#                         except Exception as e:
#                             print(f'[\u274C] Exception while writing frame to the video: {e}')
#                             continue
#                 except Exception as e:
#                     print(f'[\u274C] Exception occurred: {e}')
#                     continue

def pad_to_max_shape(images):
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    
    padded_images = []
    for image in images:
        height, width = image.shape[:2]
        top = (max_height - height) // 2
        bottom = max_height - height - top
        left = (max_width - width) // 2
        right = max_width - width - left
        
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_images.append(padded_image)
    
    return np.array(padded_images)

# Function to pad image
def pad_image(image):
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    return padded_image, pads

# Function to process a frame and save the mask
def process_frame(frame, model, transform, output_dir, object_index, cap):
    import torch
    # Apply transform and padding to frame
    transformed_frame = transform(image=frame)["image"]
    padded_frame, pads = pad_image(transformed_frame)
    x = torch.unsqueeze(tensor_from_rgb_image(padded_frame), 0)

    # Apply model to frame
    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)

    # Remove padding from mask
    unpad_mask = unpad(mask, pads)

    # Save mask to output directory
    mask_folder = os.path.join(output_dir, f"mask{object_index}")
    os.makedirs(mask_folder, exist_ok=True)
    mask_filename = os.path.join(mask_folder, f"mask_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):05d}.png")
    cv2.imwrite(mask_filename, unpad_mask * 255)

# Function to save frames to a specific folder
def save_frame(frame, output_dir, object_index, cap):
    frame_folder = os.path.join(output_dir, f"frames{object_index}")
    os.makedirs(frame_folder, exist_ok=True)
    frame_filename = os.path.join(frame_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):05d}.png")
    cv2.imwrite(frame_filename, frame)

