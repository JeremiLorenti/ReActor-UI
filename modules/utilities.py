import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
import logging  # Import logging module
from pathlib import Path
from typing import List, Any, Tuple
from tqdm import tqdm
import json
import uuid  # Import uuid for unique filenames

import modules.globals

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'
OUTPUT_DIRECTORY = 'Output'  # Define the output directory

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    # Ensure log_level has a default value if not set
    log_level = getattr(modules.globals, 'log_level', 'info')
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-loglevel', log_level]
    commands.extend(args)
    
    # Log the FFmpeg command for debugging
    try:
        logger.info(f"Running FFmpeg command: {' '.join(commands)}")
    except TypeError as e:
        logger.error(f"FFmpeg command contains non-string elements: {e}")
        return False
    
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.output.decode()}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    return False


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    try:
        output = subprocess.check_output(command).decode().strip().split('/')
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception as e:
        logger.error(f"Failed to detect FPS: {str(e)}")
    return 30.0


def extract_frames(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(['-i', target_path, '-pix_fmt', 'rgb24', os.path.join(temp_directory_path, '%04d.png')])


def create_video(target_path: str, fps: float = 30.0) -> None:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    
    logger.debug(f"Temp output path: {temp_output_path}")
    logger.debug(f"Temp directory path: {temp_directory_path}")
    
    # Ensure the temp directory exists
    if not os.path.exists(temp_directory_path):
        logger.error(f"Temp directory {temp_directory_path} does not exist")
        return

    frame_paths = get_temp_frame_paths(target_path)
    if not frame_paths:
        logger.error(f"No frames found in {temp_directory_path}. Video creation aborted.")
        return

    # Generate a unique filename for the output video
    unique_filename = f"{uuid.uuid4()}.mp4"
    unique_output_path = os.path.join(OUTPUT_DIRECTORY, unique_filename)
    
    # Ensure the Output directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    success = run_ffmpeg([
        '-r', str(fps),
        '-i', os.path.join(temp_directory_path, '%04d.png'),
        '-c:v', modules.globals.video_encoder or 'libx264',  # Fallback to 'libx264' if None
        '-crf', str(modules.globals.video_quality or 18),      # Fallback to 18 if None
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-y', unique_output_path
    ])
    if not success:
        logger.error(f"Failed to create video at {unique_output_path}")
    else:
        logger.info(f"Video created successfully at {unique_output_path}")


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if not os.path.isfile(temp_output_path):
        logger.error(f"Temp video file {temp_output_path} does not exist for audio restoration")
        return

    # Generate a unique filename for the output video with audio
    unique_filename = f"{uuid.uuid4()}.mp4"
    unique_output_path = os.path.join(OUTPUT_DIRECTORY, unique_filename)
    
    # Ensure the Output directory exists
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    done = run_ffmpeg([
        '-i', temp_output_path,
        '-i', target_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y', unique_output_path
    ])
    if not done:
        logger.error(f"Failed to restore audio to {unique_output_path}")
    else:
        logger.info(f"Audio restored successfully to {unique_output_path}")


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.png')))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if not output_path:
        output_path = 'Output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if is_video(target_path):
            target_extension = '.mp4'
        return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        # Generate a unique filename for the final output
        unique_filename = f"{uuid.uuid4()}.mp4"
        unique_output_path = os.path.join(OUTPUT_DIRECTORY, unique_filename)
        
        # Ensure the Output directory exists
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        
        if os.path.isfile(unique_output_path):
            os.remove(unique_output_path)
        shutil.move(temp_output_path, unique_output_path)
        logger.info(f"Moved temp file to {unique_output_path}")
    else:
        logger.error(f"Temp file {temp_output_path} does not exist")


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url) # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size)) # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


LAST_USED_PATHS_FILE = 'last_used_paths.json'

def save_last_used_paths(source_path: str, target_path: str) -> None:
    paths = {
        'source_path': source_path,
        'target_path': target_path
    }
    with open(LAST_USED_PATHS_FILE, 'w') as file:
        json.dump(paths, file)

def load_last_used_paths() -> Tuple[str, str]:
    if os.path.exists(LAST_USED_PATHS_FILE):
        with open(LAST_USED_PATHS_FILE, 'r') as file:
            paths = json.load(file)
            return paths.get('source_path', ''), paths.get('target_path', '')
    return '', ''