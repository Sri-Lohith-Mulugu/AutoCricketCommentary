# 🏏 Automatic Cricket Commentary System

An AI-powered full-stack web application that watches cricket videos, identifies the batsman's shot using a custom Video CNN, and automatically generates live commentary using a fine-tuned GPT-2 language model — complete with text-to-speech audio output.

---

## 🎯 What It Does

Upload a cricket video → CNN identifies the shot → GPT-2 generates commentary → Audio output delivered

---

## 🖥️ Demo

> Upload a cricket video and the system:
> - 🎥 Extracts and analyzes 32 frames from the video
> - 🏏 Identifies the cricket shot (e.g. "cover drive", "pull shot")
> - 📝 Generates realistic commentary using GPT-2
> - 🔊 Converts commentary to speech audio

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js, HTML, CSS |
| AI/ML Backend | Python, Flask, PyTorch |
| Video Processing | OpenCV (cv2) |
| Shot Classification | Custom Video CNN (PyTorch) |
| Commentary Generation | Fine-tuned GPT-2 (Hugging Face) |
| Text to Speech | Google Text-to-Speech (gTTS) |
| App Backend | Node.js, Express.js |
| Database | MongoDB |

---

## 🧠 AI Pipeline

### Step 1 — Video Processing (OpenCV)
- Extracts **32 frames** evenly distributed across the video
- Resizes each frame to **224×224** pixels
- Normalizes pixel values for CNN input

### Step 2 — Shot Classification (Video CNN)
A custom 5-layer CNN trained on 10 cricket shot types:

```
Input Video (32 frames × 224×224)
        ↓
Frame-by-frame CNN feature extraction
        ↓
Conv Layer 1 (32 filters) + ReLU + MaxPool
        ↓
Conv Layer 2 (64 filters) + ReLU + MaxPool
        ↓
Conv Layer 3 (128 filters) + ReLU + MaxPool
        ↓
Conv Layer 4 (256 filters) + ReLU + MaxPool
        ↓
Conv Layer 5 (512 filters) + ReLU + MaxPool
        ↓
Aggregate features across all frames
        ↓
Fully Connected (1024 neurons) + Dropout
        ↓
Classify into 10 shot types
```

### Step 3 — Commentary Generation (GPT-2)
- Detected shot is used as a **text prompt** for GPT-2
- Fine-tuned GPT-2 generates **realistic cricket commentary**
- Example: `"cover shot description"` → *"The batsman drives through the covers with perfect timing..."*

### Step 4 — Text to Speech
- Generated commentary converted to audio using gTTS
- Audio file returned to frontend for playback

---

## 🏏 Supported Cricket Shots (10 Classes)

| Shot | Shot | Shot |
|------|------|------|
| Cover Drive | Defense | Flick |
| Hook | Late Cut | Lofted |
| Pull | Square Cut | Straight Drive |
| Sweep | | |

---

## 📁 Project Structure

```
AutoCricketCommentary/
│
├── client/                  # React.js frontend
│   ├── src/                 # React components
│   ├── public/              # Static assets
│   └── package.json
│
├── flask_server/            # Python AI backend
│   ├── app.py               # Flask API (main entry)
│   ├── app1.py              # Video CNN model definition
│   ├── app_temp.py          # Full pipeline (CNN + GPT-2 + TTS)
│   ├── audio.py             # Text-to-speech module
│   └── working/             # Fine-tuned GPT-2 model files
│
├── server/                  # Node.js backend
│   ├── controllers/         # Route controllers
│   ├── middlewares/         # Auth middlewares
│   ├── model/               # MongoDB schemas
│   ├── routes/              # API routes
│   └── index.js             # Server entry point
│
└── README.md
```
## ⚠️ Note on Model Files

The trained model files are not included in this repository due to storage limitations:

- `video_cnn_epoch_34.pth` — Trained Video CNN weights (~500MB+)
- `working/gpt2-finetuned/` — Fine-tuned GPT-2 model files (~1GB+)

These models were trained on Google Colab. To run the full pipeline locally, you would need to retrain the models or request the weights from the project contributors.

The code for the complete pipeline is available in `flask_server/app_temp.py`.
---

## ⚙️ How It Works

```
User uploads cricket video via React frontend
        ↓
Node.js server handles authentication (MongoDB)
        ↓
Video sent to Flask API (/predict endpoint)
        ↓
OpenCV extracts 32 frames from video
        ↓
Video CNN classifies the cricket shot
        ↓
Shot label used as GPT-2 prompt
        ↓
GPT-2 generates cricket commentary text
        ↓
gTTS converts text to speech audio
        ↓
Commentary + audio returned to frontend
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js v18+
- MongoDB
- See [SETUP.md](SETUP.md) for detailed installation guide

### 1. Clone the repository
```bash
git clone https://github.com/Sri-Lohith-Mulugu/AutoCricketCommentary.git
cd AutoCricketCommentary
```

### 2. Start MongoDB
```bash
mongod
```

### 3. Start the Flask AI Server
```bash
cd flask_server
pip install flask flask-cors torch torchvision opencv-python transformers gtts pillow
python app.py
```

### 4. Start the Node.js Backend
```bash
cd server
npm install
npm start
```

### 5. Start the React Frontend
```bash
cd client
npm install
npm start
```

### 6. Open in browser
```
http://localhost:3000
```

---

## 🔌 API Reference

### POST `/predict`
Accepts a cricket video and returns generated commentary.

**Request:**
- `video` — cricket video file (mp4, avi, etc.)

**Response:**
```json
{
  "message": "The batsman plays a magnificent cover drive, timing it to perfection through the off side..."
}
```

---

## 🌍 Real-World Applications

- 📺 **Sports Broadcasting** — automate commentary for local/amateur cricket matches
- 📱 **Cricket Apps** — instant AI commentary for user-uploaded videos
- 🎓 **Cricket Coaching** — identify and label shot types automatically
- 🎮 **Cricket Games** — dynamic AI commentary generation

---

## 🧠 What I Learned

- Building and training a custom **Video CNN** for action recognition using PyTorch
- Processing video frame-by-frame using **OpenCV**
- Fine-tuning **GPT-2** for domain-specific text generation
- Integrating **text-to-speech** output using gTTS
- Connecting multiple AI models in a single pipeline
- Full stack development with React, Flask, Node.js and MongoDB

---

## 👥 Project Info

Developed as a group academic project exploring computer vision and natural language generation.

---

## 📄 License

This project is for educational purposes.
