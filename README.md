# **Text2Video Synopsis**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1taxvdTp_r2x1qt7i0lWt8BnM8AyfLY4c?usp=sharing)

This project is an advanced video analysis tool that generates comprehensive video synopses by leveraging state-of-the-art computer vision techniques. It provides a powerful solution for intelligent video summarization, particularly useful in surveillance, content analysis, and object tracking scenarios. It uses **OWL-ViT** or **Florence 2** for object detection, **SAM** for segmentation, and a custom video synopsis algorithm to produce optimized outputs.



## **Table of Contents**  
1. [Colab Notebook](#colab-notebook)  
2. [Installation](#installation)  
3. [Usage](#usage)  
   - Running Streamlit App  
   - Running Main Script  
   - Parameters and Examples  
4. [Features](#features)  
5. [To-Do](#to-do)  
6. [Related Work](#related-work)  



## **Colab Notebook**  
Run the project using this [Google Colab Notebook](https://colab.research.google.com/drive/1taxvdTp_r2x1qt7i0lWt8BnM8AyfLY4c?usp=sharing).



## **Installation**  
To install all dependencies, run:  
```bash
pip install -r requirements.txt
```



## **Usage**

### **1. Streamlit App**  
To interactively run the project on a Streamlit-based web UI:  
```bash
streamlit run ./app.py & npx localtunnel --port 8501
```  
This will expose the Streamlit app through a localtunnel link.  



### **2. Running the Main Script**  
Run `main.py` with the following example:  
```bash
python main.py \
  --input_model "OWL-ViT" \
  --video "/content/text2video_synopsis/all_rush_video.mp4" \
  --classes "people,person" \
  --epoch 100
```

#### **Parameters and Examples**  
- **`--input_model`**: Detection model to use (`OWL-ViT` or `Florence-2-large`).  
- **`--video`**: Path to the input video file.  
- **`--classes`**: Object classes to detect.  
   - For Florence: Provide a prompt sentence, e.g.,  
     - Simple ones `"People in the video" , "Car on the road"`  
     - Complex ones `"People with black t-shirt" , "People with suitcase"`  
   - For OWL-ViT: Provide an **OPEN_VOCABULARY_DETECTION** comma-separated classes, e.g.,  
     - `"car,person,dog"`  
- **`--epoch`**: Number of iterations for video synopsis optimization.



## **Features**  
1. **Motion Detection**: Focuses processing on video segments with significant motion.  
2. **Object and Action Detection**: Uses state-of-the-art models like Florence and OWL-ViT for object detection, and SAM for segmentation.  
3. **Flexible Synopsis Generation**: Creates optimized video summaries based on user-defined object criteria
4. **Versatile Use Cases**:

   - Surveillance video summarization
   - Targeted object tracking
   - Intelligent video content analysis



## **To-Do**  
- [x] Web UI (Streamlit App)  
- [ ] Robust Video Synopsis
- [ ] Add diagram explaining the project - input(multiple images showing 24 hour cctv video) output(5 frames of output video and adding a gif for the same)



## **Related Work**  
- [Video Synopsis](https://github.com/mithunparab/video_synopsis)  
- [Microsoft Florence-2 Large](https://huggingface.co/microsoft/Florence-2-large)  
- [Ultralytics](https://github.com/ultralytics)  
