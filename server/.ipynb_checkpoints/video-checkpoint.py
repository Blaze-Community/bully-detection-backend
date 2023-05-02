import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

__model = None
UPLOAD_FOLDER = 'static/uploads/'
__class_list = ["violence", "non-violence"]


def classify_video(video_file_path, SEQUENCE_LENGTH = 16, IMAGE_HEIGHT = 112, IMAGE_WIDTH = 112):
    
    result = []
    
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Read a frame.
        success, frame = video_reader.read() 

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        frames_list.append(cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)))

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = __model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = __class_list[predicted_label]
    
    # Saved the predicted action along with the prediction confidence.
    result.append({
        'action_predicted' : str(predicted_class_name),
        'confidence' : str(predicted_labels_probabilities[predicted_label])
    })
        
    # Release the VideoCapture object. 
    video_reader.release()
    os.remove(video_file_path)
    return result

def upload_video(file):
    filename = secure_filename(file.filename)
    video_path = os.path.join("../" + UPLOAD_FOLDER, filename)
    file.save(video_path)
    return video_path

def load_saved_video_artifacts():
    print("loading saved video artifacts...start")
    
    global __model 
    if __model is None:
        __model = tf.keras.models.load_model('./artifacts/c3d_model___Date_Time_2022_04_04__07_03_33.h5')
        
    print("loading saved video artifacts...done")
        
    
if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_video('../static/uploads/3.mp4'))