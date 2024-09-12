from flask import Flask, request, jsonify, render_template, Response, send_file
from flask_cors import CORS
import os
import shutil
import uuid
import threading
import time
import queue
import sys
import logging
import json
import modules.core as core

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        log_entry = self.format(record)
        self.queue.put(log_entry)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Queue handler
queue_handler = QueueHandler(log_queue)
queue_handler.setLevel(logging.DEBUG)
queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(queue_handler)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    source_file = request.files['source_file']
    target_files = request.files.getlist('target_files')
    output_dir = request.form['output_dir']
    face_enhancer = 'face_enhancer' in request.form
    keep_fps = 'keep_fps' in request.form
    keep_audio = 'keep_audio' in request.form
    keep_frames = 'keep_frames' in request.form
    many_faces = 'many_faces' in request.form
    max_memory = int(request.form.get('max_memory', core.suggest_max_memory()))
    execution_provider = request.form.getlist('execution_provider')
    execution_threads = int(request.form.get('execution_threads', core.suggest_execution_threads()))

    source_path = os.path.join(UPLOAD_FOLDER, source_file.filename)
    source_file.save(source_path)

    target_paths = []
    for target_file in target_files:
        target_path = os.path.join(UPLOAD_FOLDER, target_file.filename)
        target_file.save(target_path)
        target_paths.append(target_path)

    def background_process():
        total_files = len(target_paths)
        output_files = []
        for i, target_path in enumerate(target_paths):
            # Set global variables
            core.modules.globals.source_path = source_path
            core.modules.globals.target_path = target_path

            # Generate a unique name for the output file
            if core.modules.utilities.is_video(target_path):
                output_filename = f"{uuid.uuid4()}.mp4"
            else:
                output_filename = f"{uuid.uuid4()}.png"
            core.modules.globals.output_path = os.path.join(output_dir, output_filename)

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            core.modules.globals.frame_processors = ['face_swapper']
            if face_enhancer:
                core.modules.globals.frame_processors.append('face_enhancer')
            core.modules.globals.keep_fps = keep_fps
            core.modules.globals.keep_audio = keep_audio
            core.modules.globals.keep_frames = keep_frames
            core.modules.globals.many_faces = many_faces
            core.modules.globals.max_memory = max_memory
            core.modules.globals.execution_providers = core.decode_execution_providers(execution_provider)
            core.modules.globals.execution_threads = execution_threads

            # Start processing
            core.start()

            # Add output file to the list
            output_files.append(core.modules.globals.output_path)

            # Update progress
            progress = ((i + 1) / total_files) * 100
            log_queue.put(json.dumps({"progress": progress}))
        
        # Send completion message with output files
        log_queue.put(json.dumps({"status": "complete", "output_files": output_files}))

    threading.Thread(target=background_process).start()

    return jsonify({"status": "Processing started", "output_dir": output_dir})

@app.route('/status', methods=['GET'])
def status():
    def generate():
        while True:
            message = log_queue.get()
            yield f"data: {message}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_file(filename, as_attachment=True)

@app.route('/preview/<path:filename>', methods=['GET'])
def preview_file(filename):
    return send_file(filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)