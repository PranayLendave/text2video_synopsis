import cv2
import os
import datetime

def create_video_from_frames(frames_directory):
    timestamp = datetime.datetime.now()
    video_name = "{}.avi".format(timestamp.strftime("%Y_%m_%d-%H_%M_%S"))

    # Get the list of frames in the directory
    frames = sorted(os.listdir(frames_directory))
    # print(os.path.join(frames_directory, frames[0]))

    # Read the first frame to get the frame size
    first_frame = cv2.imread(os.path.join(frames_directory, frames[0]))
    frame_height, frame_width, _ = first_frame.shape

    # Define the video writer
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))

    # Write each frame to the video
    for frame_file in frames:
        frame_path = os.path.join(frames_directory, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    # Release the video writer and destroy any OpenCV windows
    video.release()
    cv2.destroyAllWindows()

# Example usage
frames_directory = './synopsis_frames/'
create_video_from_frames(frames_directory)
