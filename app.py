from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
from utils.video_swin import predict_video, initialize
import mimetypes

app = Flask(__name__)

CORS(app)
initialize()

# Set upload folder inside the static directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return " Fight Detection Flask App is running!"

@app.route('/uploads/<filename>')
def serve_video(filename):
    try:
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Attempting to serve video from: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Video not found at: {video_path}")
            return jsonify({'error': 'Video not found'}), 404

        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type:
            mime_type = 'video/mp4'  # Default to mp4 if type can't be determined
        print(f"Detected MIME type: {mime_type}")

        # Enable range requests for video streaming
        range_header = request.headers.get('Range', None)
        if not range_header:
            print("No range header, serving full file")
            return send_from_directory(UPLOAD_FOLDER, filename, mimetype=mime_type)

        # Handle range requests for video streaming
        size = os.path.getsize(video_path)
        start, end = 0, size - 1

        try:
            range_header = range_header.replace('bytes=', '')
            start_bytes, end_bytes = range_header.split('-')
            start = int(start_bytes) if start_bytes else 0
            end = int(end_bytes) if end_bytes else size - 1
        except ValueError:
            print("Invalid range header format, serving full file")
            return send_from_directory(UPLOAD_FOLDER, filename, mimetype=mime_type)

        chunk_size = end - start + 1
        with open(video_path, 'rb') as f:
            f.seek(start)
            data = f.read(chunk_size)

        response = Response(
            data,
            206,
            mimetype=mime_type,
            direct_passthrough=True,
            content_type=mime_type
        )
        response.headers.add('Content-Range', f'bytes {start}-{end}/{size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(chunk_size))
        
        return response
    except Exception as e:
        print(f"Error serving video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    video_file = request.files['video']
    print(f"Received video file: {video_file.filename}")
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(save_path)
    print(f"Saved video to: {save_path}")

    result = predict_video(save_path)
    print("Prediction result:", result)
    return jsonify(result)

@app.route('/uploads', methods=['GET'])
def list_uploads():
    try:
        files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        print(f"Found {len(files)} video files in {UPLOAD_FOLDER}")
        return jsonify({'files': files})
    except Exception as e:
        print(f"Error listing uploads: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<filename>')
def get_prediction(filename):
    try:
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404
        
        result = predict_video(video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting server with upload folder: {UPLOAD_FOLDER}")
    app.run(debug=True)
