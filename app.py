import streamlit as st
import os
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
import shutil
from main import main
from supplementary.our_args import args

# Setting the page configuration with an icon
image_directory = "./supplementary/vs_clip.png"
image = Image.open(image_directory)
PAGE_CONFIG = {
    "page_title": "Video Synopsis",
    "page_icon": image,
    "layout": "wide",
    "initial_sidebar_state": "auto"
}
st.set_page_config(**PAGE_CONFIG)

def setup_environment(args):
    # Path settings and directory cleanup
    output_path = args["output"]
    optimized_tubes_dir = "optimized_tubes"
    for path in [output_path, optimized_tubes_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    os.chdir(output_path)

    # Video capture and background preparation
    cap = cv2.VideoCapture(args['video'])
    cap1 = cv2.VideoCapture(args['video'])
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    frame_width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    bgimg = prepare_background_image(cap1, fps)
    print(f'[original video] frame_width: {frame_width}, frame_height: {frame_height} \u2705')
    print(f'[original video] Total frames: {video_length} \u2705')
    print(f'[original video] FPS: {fps} \u2705')
    return cap, video_length, bgimg, fps

def prepare_background_image(cap, fps):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rand_indices = np.random.choice(total_frames, size=fps, replace=False)
    Sframes = [
        cap.read()[1] 
        for _ in rand_indices 
        if cap.set(cv2.CAP_PROP_POS_FRAMES, _)
    ]

    if Sframes:
        median_frame = np.median(np.array(Sframes), axis=0).astype(np.uint8)

        # Ensure the parent directory of bg_path exists
        bg_path = args['bg_path']
        os.makedirs(os.path.dirname(bg_path), exist_ok=True)

        cv2.imwrite(bg_path, median_frame)
        bgimg = cv2.cvtColor(np.asarray(Image.open(bg_path)), cv2.COLOR_RGB2BGR)
        return bgimg
    else:
        raise ValueError("[Error]: Unable to calculate median frame. No valid frames were sampled.")


def run_main(args):
    cap, video_length, bgimg, fps = setup_environment(args)
    final = args['masks']
    temp_video_name = os.path.abspath(f"../{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_temp.mp4")
    final_video_name = temp_video_name.replace('_temp', '')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(temp_video_name, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    main(args, cap, video, video_length, final, bgimg, args['energy_opt'], args['epochs'], final_video_name)
    
    cap.release()
    video.release()

    if os.path.exists(temp_video_name):
        command = f'ffmpeg -loglevel error -i "{temp_video_name}" -vcodec libx264 -crf 23 -preset fast "{final_video_name}"'
        os.system(command)
        if os.path.exists(final_video_name):
            os.remove(temp_video_name)
            st.video(final_video_name)
        else:
            st.error('Failed to process video correctly with FFmpeg. \u274C')
    else:
        st.error(f'[Info] Video file not found at {temp_video_name} \u274C')

def handle_file_upload(uploaded_file):
    if uploaded_file:
        filename = os.path.abspath(uploaded_file.name)
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return filename

# Video Synopsis Configuration Form
with st.form("input_form"):
    # Title
    st.title("🎬 Video Synopsis Configuration")
    
    # File Uploader Section
    st.header("Step 1: Upload Your Video")
    uploaded_file = st.file_uploader("Choose a video file (MP4/AVI)", type=['mp4', 'avi'])
    video_path = handle_file_upload(uploaded_file) if uploaded_file else None
    
    # Display Uploaded Video
    if video_path:
        st.markdown("---")  # Divider for clarity
        st.subheader("Preview of Uploaded Video")
        _, video_col, _ = st.columns([1, 8, 1])  # Centering the video display
        with video_col:
            st.video(video_path, format="video/mp4", start_time=0)
        st.markdown("---")

    # Configuration Columns
    st.header("Step 2: Configure Detection Settings")
    col1, col2, col3 = st.columns(3)

    # Column 1: Model Selection and Input
    with col1:
        st.subheader("🔍 Detection Model")
        input_model = st.selectbox(
            "Select Model:",
            ["OWL-ViT", "Florence-2-large"],
            help="OWL-ViT uses classes (e.g., 'person,car'), while Florence-2-large supports descriptive sentences."
        )

        classes = st.text_input(
            "Enter Detection Query:",
            value="people,people",
            help="For OWL-ViT: Enter comma-separated classes. For Florence-2-large: Use a descriptive sentence."
        )

    # Column 2: Optimization Settings
    with col2:
        st.subheader("⚙️ Optimization Settings")
        energy_opt = st.checkbox("Enable Energy Optimization", value=True)
        epochs = st.number_input(
            "Number of Epochs:",
            value=1000, 
            min_value=1, 
            format="%d",
            help="Set the number of iterations for processing (higher values improve accuracy)."
        )

    # Column 3: Output Settings
    with col3:
        st.subheader("📁 Output Configuration")
        ext = st.text_input(
            "Object Extraction Format:",
            value='.png',
            help="Specify the file extension for extracted objects (e.g., .png, .jpg)."
        )
        dvalue = st.number_input(
            "Compression Level:",
            value=9, 
            min_value=0, 
            max_value=9, 
            format="%d",
            help="Choose a compression level (0 for no compression, 9 for maximum)."
        )

    # Display Selected Options for Verification
    st.markdown("---")
    st.subheader("📝 Summary of Selections")
    st.write(f"**Model:** {input_model}")
    st.write(f"**Detection Query:** {classes}")
    st.write(f"**Energy Optimization:** {'Enabled' if energy_opt else 'Disabled'}")
    st.write(f"**Epochs:** {epochs}")
    st.write(f"**Output Format:** {ext}")
    st.write(f"**Compression Level:** {dvalue}")

    # Submit Button
    submitted = st.form_submit_button("🚀 Run Configuration")
    if submitted:
        st.success("Configuration Submitted! Processing will begin shortly.")


if submitted and video_path:
    if input_model == "OWL-ViT":
      classes = [c.strip() for c in classes.split(',') if c.strip()]
    # Update the args dynamically without redefining
    args.update({
        'video': video_path,
        # 'buff_size': buff_size,
        'input_model': input_model,
        'classes': classes,
        'ext': ext,
        'dvalue': dvalue,
        'energy_opt': energy_opt,
        'epochs': epochs
    })

    output_video = run_main(args)
    if output_video:
        _, right_col, _ = st.columns([1, 8, 1])
        with right_col:
            st.subheader("Video Synopsis")
            st.video(output_video)
else:
    if submitted:
        st.error("Please upload a video file to proceed. \u274C")
