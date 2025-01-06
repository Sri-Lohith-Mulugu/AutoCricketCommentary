from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

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
        # Save the file or process it here
        # video.save(os.path.join('uploads', video.filename))
        return jsonify({'message': 'Video Received'}), 200
    else:
        return jsonify({'message': 'Invalid file type. Please upload a video.'}), 400

if __name__ == "__main__":
    app.run(debug=True)
