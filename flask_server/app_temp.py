import torch
import torch.nn as nn
from torchvision import transforms
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import subprocess  # To run audio.py script

# Define the same transformations as used during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to load and preprocess a video
def load_video(video_path, num_frames=32, transform=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    step = max(1, total_frames // num_frames)
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))  # Resize for CNN
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()

    # Ensure we have the required number of frames
    frames = frames[:num_frames] + [frames[-1]] * (num_frames - len(frames))  # Padding with the last frame if needed
    return torch.stack(frames)

# Define the VideoCNN model
class VideoCNN(nn.Module):
    def __init__(self, num_classes):
        super(VideoCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(512 * 7 * 7, 1024)  # Adjust based on CNN output size
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)  # Flatten frame dimension

        frame_features = self.cnn(x)
        frame_features = frame_features.view(frame_features.size(0), -1)  # Flatten features
        frame_features = frame_features.view(batch_size, num_frames, -1).mean(dim=1)  # Aggregate across frames

        x = torch.relu(self.fc1(frame_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Load the model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "video_cnn_epoch_34.pth"
model = VideoCNN(num_classes=10)  # Update with the actual number of classes
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Load the fine-tuned GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("/working/gpt2-finetuned")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("/working/gpt2-finetuned")

# Function for generating text
def generate_text(prompt, model, tokenizer, max_length=40):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to predict the output label for a video
def predict_video(video_path, class_names):
    frames = load_video(video_path, num_frames=32, transform=transform)
    frames = frames.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(frames)
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        _, predicted_class = torch.max(probabilities, 1)

    return class_names[predicted_class.item()], probabilities.squeeze().cpu().numpy()

# Flask server
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'message': 'No video file part'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if video and video.content_type.startswith('video/'):
        # Save the video temporarily for processing
        temp_video_path = os.path.join('temp', video.filename)
        os.makedirs('temp', exist_ok=True)
        video.save(temp_video_path)

        # Class names for prediction
        class_names = ["cover", "defense", "flick", "hook", "late_cut", "lofted", "pull", "square_cut", "straight", "sweep"]

        try:
            predicted_class, probabilities = predict_video(temp_video_path, class_names)
            prompt = f"{predicted_class} shot description"
            generated_text = generate_text(prompt, gpt2_model, gpt2_tokenizer)
            print(f"Generated Text: {generated_text}")
  # Clean up temporary file
            return jsonify({
                'message': generated_text
            }), 200

        except Exception as e:
            os.remove(temp_video_path)  # Clean up temporary file
            return jsonify({
                'message': 'Error processing video',
                'error': str(e) 
            }), 500

    else:
        return jsonify({'message': 'Invalid file type. Please upload a video.'}), 400

if __name__ == "__main__":
    app.run(debug=True)
