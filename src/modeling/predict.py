import os
from pathlib import Path
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger

# Set up paths
MODELS_DIR = Path('/content/drive/MyDrive/big data/')
TEST_VID_DIR = Path('/content/drive/MyDrive/big data/test vid/')

# Load the trained model
model_path = MODELS_DIR / 'model_at_epoch_2.h5'
model = load_model(model_path)

def preprocess_video(file_path, resize=(224, 224), target_frames=64):
    cap = cv2.VideoCapture(file_path)
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(len_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    frames = np.array(frames)
    flows = getOpticalFlow(frames)

    result = np.zeros((len(flows), 224, 224, 5))
    result[..., :3] = frames
    result[..., 3:] = flows

    result = uniform_sampling(result, target_frames)
    result[..., :3] = normalize(result[..., :3])
    result[..., 3:] = normalize(result[..., 3:])

    return result[np.newaxis, ...]

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def uniform_sampling(video, target_frames=64):
    len_frames = len(video)
    interval = int(np.ceil(len_frames / target_frames))
    sampled_video = []

    for i in range(0, len_frames, interval):
        sampled_video.append(video[i])

    num_pad = target_frames - len(sampled_video)
    padding = [video[0]] * num_pad
    sampled_video.extend(padding)

    return np.array(sampled_video, dtype=np.float32)

def getOpticalFlow(video):
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (224, 224, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow)

    flows.append(np.zeros((224, 224, 2)))

    return np.array(flows, dtype=np.float32)

def evaluate_model(model, test_folder):
    labels = {'normal': 0, 'restrict': 1}
    y_true = []
    y_pred = []

    for subfolder in os.listdir(test_folder):
        subfolder_path = test_folder / subfolder
        if subfolder_path.is_dir():
            for video_file in os.listdir(subfolder_path):
                video_path = subfolder_path / video_file
                if video_path.suffix == '.mp4':
                    preprocessed_video = preprocess_video(str(video_path))
                    predictions = model.predict(preprocessed_video)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    y_pred.append(predicted_class)
                    y_true.append(labels[subfolder])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

# Main evaluation
if __name__ == "__main__":
    # Configure logging
    logger.add("/content/drive/MyDrive/big data/evaluation.log", rotation="500 MB", level="INFO")

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, TEST_VID_DIR)

    # Logging results
    logger.info(f'Evaluation results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')