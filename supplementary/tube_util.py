import cv2
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def load_new_times(directory):
    new_times = defaultdict(dict)
    for filename in sorted(glob.glob(f"{directory}/*.txt")):
        with open(filename, "r") as file:
            for line in file:
                Tube, n, _, _, _, _, curr_time, *_  = line.strip().split(',')
                new_times[int(Tube)][int(n)] = float(curr_time)
    return new_times

def blend_roi_on_background(background, roi, roi_mask, x1, x2, y1, y2, clear=False):
    if x1 < 0 or y1 < 0 or x2 > background.shape[1] or y2 > background.shape[0]:
        print("ROI is out of the bounds of the background image. Clamping to bounds.")
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, background.shape[1]), min(y2, background.shape[0])

    if x2 <= x1 or y2 <= y1:
        print("Adjusted ROI coordinates result in non-positive area.")
        return background  # Return unchanged background if ROI area is non-positive

    roi_width, roi_height = x2 - x1, y2 - y1
    roi_resized = cv2.resize(roi, (roi_width, roi_height))
    mask_resized = cv2.resize(roi_mask, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

    # Convert mask to binary and ensure it is CV_8U
    _, mask_binary = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)
    if mask_binary.dtype != np.uint8:
        mask_binary = mask_binary.astype(np.uint8)

    roi_region = background[y1:y2, x1:x2]
    if roi_resized.shape[:2] != roi_region.shape[:2]:
        print("ROI resized shape does not match the ROI region shape.")
        return background

    # Apply the mask
    mask_3ch = cv2.merge([mask_binary] * 3)  # Ensure mask is 3 channels if needed
    roi_fg = cv2.bitwise_and(roi_resized, roi_resized, mask=mask_binary)
    roi_bg = cv2.bitwise_and(roi_region, roi_region, mask=cv2.bitwise_not(mask_binary))
    blended_roi = cv2.add(roi_fg, roi_bg)
    background[y1:y2, x1:x2] = blended_roi

    return background

def Tube_mod(args, video, final, bgimgpath='../bg/background_img.png', dir1 = "*", dir2 = "../optimized_tubes"):
    new_times = load_new_times(dir2)
    bgimg = np.asarray(Image.open(bgimgpath))

    tube_queue = defaultdict(list)

    filenames = sorted(glob.glob(f"{dir1}/*.txt"))
    for filename in tqdm(filenames):
        with open(filename) as file:
            while (line := file.readline().rstrip('')):
                Tube, n, x1, x2, y1, y2, curr_time, *_ = line.split(',')
                Tube, n, x1, x2, y1, y2 = int(Tube), int(n), int(x1), int(x2), int(y1), int(y2)
                curr_time = new_times[Tube].get(n, float(curr_time))  # Use new time if available, else original

                tube_queue[curr_time].append((Tube, n, x1, x2, y1, y2))

    # Process tubes sorted by time
    for curr_time in sorted(tube_queue.keys()):
        combined_background = bgimg.copy() 

        for Tube, n, x1, x2, y1, y2 in tube_queue[curr_time]:
            image_path = f'{str(Tube).zfill(4)}/{str(n).zfill(4)}' + args['ext']
            mask_path = f'{str(final).zfill(4)}/{str(Tube).zfill(4)}/{str(n).zfill(4)}' + args['ext']

            image = np.asarray(Image.open(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask = np.asarray(Image.open(mask_path))

            if image.shape[:2] != mask.shape[:2]:
                print(f"ROI {image.shape[:2]} and mask {mask.shape[:2]} dimensions do not match")

            temp_clone = blend_roi_on_background(combined_background, image, mask, x1, x2, y1, y2)
            combined_background[y1:y2, x1:x2] = temp_clone[y1:y2, x1:x2]

        background = combined_background.copy()

        if Tube > 0:
            video.write(background) 