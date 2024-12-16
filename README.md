Here's a concise and well-structured README file for your project:

---

# **Text2Video Synopsis**  

A project that generates concise video synopses by detecting, segmenting, and summarizing objects in video footage. It uses **OWL-ViT** for object detection, **SAM** for segmentation, and a custom video synopsis algorithm to produce optimized outputs.

---

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

---

## **Colab Notebook**  
Run the project using this [Google Colab Notebook](#) (replace `#` with your Colab link).

---

## **Installation**  
To install all dependencies, run:  
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Streamlit App**  
To interactively run the project on a Streamlit-based web UI:  
```bash
!streamlit run ./app.py & npx localtunnel --port 8501
```  
This will expose the Streamlit app through a localtunnel link.  

---

### **2. Running the Main Script**  
Run `main.py` with the following example:  
```bash
!python main.py \
  --input_model "OWL-ViT" \
  --video "/content/text2video_synopsis/all_rush_video.mp4" \
  --classes "people,person" \
  --epoch 100
```

#### **Parameters and Examples**  
- **`--input_model`**: Detection model to use (`OWL-ViT` or `Florence-2-large`).  
- **`--video`**: Path to the input video file.  
- **`--classes`**: Provide a prompt for Object classes to detect.  
   - For Florence:  sentence, e.g.,  
     - Simple ones `"People in the video" , "Car on the road"`  
     - Complex ones `"People with black t-shirt" , "People with suitcase"`  
   - For OWL-ViT: Provide an **OPEN_VOCABULARY_DETECTION** comma-separated classes, e.g.,  
     - `"car,person,dog"`  
- **`--epoch`**: Number of iterations for video synopsis optimization.

---

## **Features**  
1. **Motion Detection**: Focuses processing on video segments with significant motion.  
2. **Object and Action Detection**: Uses state-of-the-art models like Florence and OWL-ViT for object detection, and SAM for segmentation.  
3. **Video and Mask Output**: Generates annotated video outputs along with mask videos showing detected segments.

---

## **To-Do**  
- [x] Web UI (Streamlit App)  
- [ ] Robust Video Synopsis  

---

## **Related Work**  
- [Text2Segment Video](https://github.com/mithunparab/text2segment_video)  
- [Microsoft Florence-2 Large](https://huggingface.co/microsoft/Florence-2-large)  
- [Ultralytics](https://github.com/ultralytics)  

---

This README provides an overview, clear instructions for setup and usage, and references for further exploration.
