from sort import Sort
from tube_util import Tube, Tube_mod
from supplementary.our_args import args
from energy import optimize_tube
from utils import pad_image, pad_to_max_shape
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
import cv2
import numpy as np
import os
import csv
import datetime
import torch
import shutil
import time
from PIL import Image
from tqdm import tqdm

from ultralytics import SAM
from utils import pad_image, pad_to_max_shape


# Define the function for running the model inference

def initialize_models(input_model, device):

    owl_vit_detector, florence_processor, florence_model, seg_model = None, None, None, None
    
    if input_model == "OWL-ViT":
        if owl_vit_detector is None:
            checkpoint = "google/owlvit-base-patch32"
            owl_vit_detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=device)
    elif input_model == "Florence-2-large":
        if florence_processor is None or florence_model is None:
            model_id = 'microsoft/Florence-2-large'
            florence_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
            florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    if seg_model is None:
        seg_model = SAM("sam2_b.pt")

    return owl_vit_detector, florence_processor, florence_model, seg_model 


def run_florence(model, processor, image, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>', text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def combine_masks(masks):
    # Combine all masks into a single mask
    combined_mask = np.any(masks, axis=0)
    return combined_mask.astype(float)

def main(args: dict, 
         cap: cv2.VideoCapture, 
         video: cv2.VideoWriter,  
         video_length:int, 
         final:np.ndarray, 
         bgimg:np.ndarray, 
         energy_opt:bool=True, 
         epochs:int=1000, 
         final_video_name:str=None
        ):
   
    start_time = time.time()  # Start the timer
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'[Info] Device :**{device}** \u2705')
    image_number = 1
    pbar = tqdm(total=video_length, desc=f'⏳ Processing Frames', unit='frame')
    
    owl_vit_detector, florence_processor, florence_model, seg_model = initialize_models(args["input_model"], device)

    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
    object_id_mapping = {}
    next_available_id = 1

    while True:
        pbar.update(1)
        isTrue, frame = cap.read()
        if not isTrue:
            break
        original = frame.copy()
        height, width, _ = original.shape

        # Resize the image for processing
        input_image = original
        image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        if args['input_model'] == "OWL-ViT":
            predictions = owl_vit_detector(image_pil, candidate_labels=args['classes'])
            results = [
                [detection['box']['xmin'], detection['box']['ymin'], detection['box']['xmax'], detection['box']['ymax']]
                for detection in predictions
            ]
        elif args['input_model'] == "Florence-2-large":
            input_image_fl = cv2.resize(original, (640, 480))
            input_image_fl = cv2.cvtColor(input_image_fl, cv2.COLOR_BGR2RGB)
            input_image_fl = Image.fromarray(input_image_fl)
            florence_results = run_florence(florence_model, florence_processor, image=input_image_fl, task_prompt='<CAPTION_TO_PHRASE_GROUNDING>', text_input=args['classes'])
            predictions = florence_results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
            scale_x, scale_y = width / 640, height / 480
            results = [
                [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
                for x1, y1, x2, y2 in predictions
            ]


        if results:
            seg_results = seg_model(image_pil, bboxes=results)
            masks = seg_results[0].masks.data  # This is a tensor containing all masks

            # Convert the mask tensor to a numpy array
            mask_array = masks.cpu().numpy()
            combined_mask = combine_masks(mask_array)
            mask = combined_mask.astype(np.uint8)

            mask = mask.astype(np.uint8)

            mask_e = mask # changeing this since the value is already 0-255

            detections = []
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            
            detections = results

            if detections:
                tracked_objects = tracker.update(np.array(detections))
                for track in tracked_objects:
                    objectID = int(track[4])
                    coords = [max(0, int(coord)) for coord in track[:4]]
                    if objectID not in object_id_mapping:
                        object_id_mapping[objectID] = next_available_id
                        next_available_id += 1
                    
                    new_id = object_id_mapping[objectID]
                    centroid_dict = {objectID: (*coords[:2], *coords[2:])}

                    ROI = original[coords[1]:coords[3], coords[0]:coords[2]]
                    mask_roi = mask_e[coords[1]:coords[3], coords[0]:coords[2]]
                    
                    if ROI.size != 0:
                        TubeID = str(new_id).zfill(4)
                        curr_time_str = f'{current_time:.2f}'
                        os.makedirs(TubeID, exist_ok=True)
                        os.makedirs(f'../masks/{TubeID}', exist_ok=True)
                        cv2.imwrite(f'{TubeID}/{str(image_number).zfill(4)}' + args['ext'], ROI)
                        # if mask_roi is None or mask_roi.size == 0:
                        #   print("Error: Empty or invalid image!")
                        # else:
                        #   print("Image Shape:", mask_roi.shape)
                        cv2.imwrite(f'../masks/{TubeID}/{str(image_number).zfill(4)}' + args['ext'], mask_roi)

                        filename = f'{TubeID}/{TubeID}node.txt'
                        filenamecsv = f'{TubeID}/{TubeID}node.csv'
                        with open(filename, 'a') as out:
                            out.write(f'{TubeID}, {image_number}, {coords[0]}, {coords[2]}, {coords[1]}, {coords[3]}, {curr_time_str},\n')
                        with open(filenamecsv, 'a', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            if csv_file.tell() == 0:
                                writer.writerow(['T', 'n', 'x1', 'y1', 'x2', 'y2', 'time', 'contour'])
                            writer.writerow([int(TubeID), int(image_number), *coords, curr_time_str, 0])
                        
                        image_number += 1
        
    pbar.close()
    
    if energy_opt:
        optimize_tube(files_pattern=args['files_csv_dir'], output_dir=args['optimized_tubes_dir'], video_length=video_length, epochs=epochs)
        Tube(args, video, bgimg=bgimg, final=final, dir2 = f"{args['optimized_tubes_dir']}/*.txt")
    else:
        Tube(args, video, bgimg=bgimg, final=final, dir2="*/*.txt")

    cap.release()
    print(f'[Info] Video Synopsis is saved at {final_video_name} \u2705')
    end_time = time.time()  # End the timer
    total_time = end_time - start_time  # Calculate the total time taken
    print(f'⏳ [Info] Total time taken: {total_time:.2f} seconds \u23F1')
    print(u'[Finish \U0001F64C \U0001F3C1]...')

if __name__ == "__main__":
    # Set paths
    output_path = args["output"]
    final = args["masks"]
    synopsis_frames = args["synopsis_frames"]
    energy_opt = args["energy_optimization"]
    epochs = args["epochs"]

    # Create or clear directories
    def prepare_directory(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    
    # Create or clear directories
    for path in [output_path, synopsis_frames, final]:
        prepare_directory(path)

    os.chdir(output_path)  # Change to output directory

    # Configure background subtraction
    fgbg = cv2.createBackgroundSubtractorKNN(127, cv2.THRESH_BINARY, 1)
    fgbg.setDetectShadows(False)

    # Video capture configuration
    video_path = args["video"]
    cap = cv2.VideoCapture(video_path)  # Main video capture
    cap1 = cv2.VideoCapture(video_path)  # Separate for background subtraction

    if not cap.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"[Error]: Unable to open video file {video_path}")

    # Obtain video properties
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    video_length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Output video properties
    print(f"[Original Video] Frame Width: {frame_width}, Frame Height: {frame_height} ✅")
    print(f"[Original Video] Total Frames: {video_length} ✅")
    print(f"[Original Video] FPS: {fps} ✅")

    # Random frame selection for background median
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    rand_ids = np.random.choice(total_frames, size=fps, replace=False)
    sampled_frames = []

    for frame_id in rand_ids:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap1.read()
        if ret and frame is not None:
            sampled_frames.append(frame)

    # Ensure all frames are padded to the same size
    padded_frames = pad_to_max_shape(sampled_frames)

    # Compute median frame and save it
    median_frame = np.median(padded_frames, axis=0).astype(np.uint8)
    bg_path = args["bg_path"]
    os.makedirs(os.path.dirname(bg_path), exist_ok=True)
    cv2.imwrite(bg_path, median_frame)

    # Preprocess median frame
    gray_median = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    smooth_median = cv2.GaussianBlur(gray_median, (5, 5), 0)

    # Load and prepare background image
    bgimg = np.asarray(Image.open(bg_path))
    bgimg = cv2.cvtColor(bgimg, cv2.COLOR_RGB2BGR)

    # Video writer setup
    if frame_width > 0 and frame_height > 0:
        video_name = f"../{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.mp4"
        video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )

        if not video.isOpened():
            raise RuntimeError("[Error]: Could not open video writer. ❌")
    else:
        raise ValueError("[Error]: Invalid frame dimensions. ❌")

    # Main processing
    main(args, cap, video, video_length, final, bgimg, energy_opt, epochs, video_name)
