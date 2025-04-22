from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

from Server.gemini import GEMINI_API_KEY, analyze_video_with_gemini

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max-limit

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_video():
    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    
    file = request.files['video']
    
    # If user does not select file, browser might submit an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        if not file.filename:
            return jsonify({'error': 'Filename missing'}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze video with Gemini if API key is available
        analysis_result = {"status": "Analysis skipped - no API key"}
        if GEMINI_API_KEY:
            analysis_result = analyze_video_with_gemini(file_path)
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': filename,
            'file_path': file_path,
            'size': os.path.getsize(file_path),
            'analysis': analysis_result
        }), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/', methods=['GET'])
def index():
    return '''
    <!doctype html>
    <title>Upload Video File</title>
    <h1>Upload Video File</h1>
    <form method=post action=/upload enctype=multipart/form-data>
      <input type=file name=video>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
