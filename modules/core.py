import os
import sys
import queue
import json
import logging
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

import modules.globals
import modules.metadata
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch
else:
    modules.globals.execution_providers = ['cuda']  # Default to CUDA if available

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Set up logging
logger = logging.getLogger(__name__)

status_messages = queue.Queue()

def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16

def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available_providers = onnxruntime.get_available_providers()
    logger.info(f"Available providers: {available_providers}")
    selected_providers = [provider for provider, encoded_execution_provider in zip(available_providers, encode_execution_providers(available_providers))
                          if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
    logger.info(f"Selected providers: {selected_providers}")
    return selected_providers

def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()

def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cuda'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')
    program.add_argument('--log-level', help='Logging level', dest='log_level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.video_encoder = 'libx264'  # Default video encoder
    modules.globals.video_quality = 18  # Default video quality
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    logger.info(f"Using execution providers: {modules.globals.execution_providers}")
    modules.globals.execution_threads = args.execution_threads
    modules.globals.log_level = args.log_level  # Set the log_level in globals

    #for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False
    
    modules.globals.nsfw = False

    # translate deprecated args
    if args.source_path_deprecated:
        logger.warning('Argument -f and --face are deprecated. Use -s and --source instead.')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)

def start() -> None:
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw == False:
            from modules.predicter import predict_image
            if predict_image(modules.globals.target_path):
                destroy()
        shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status(f'Progressing with {frame_processor.NAME}...')
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        open_output(modules.globals.output_path)  # Display the output
        return
    # process image to videos
    if modules.globals.nsfw == False:
        from modules.predicter import predict_video
        if predict_video(modules.globals.target_path):
            destroy()
    update_status('Creating temp resources...')
    create_temp(modules.globals.target_path)
    update_status('Extracting frames...')
    extract_frames(modules.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    total_frames = len(temp_frame_paths)
    for i, frame_processor in enumerate(get_frame_processors_modules(modules.globals.frame_processors)):
        update_status(f'Progressing with {frame_processor.NAME}...')
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()
        # Update progress
        progress = ((i + 1) / total_frames) * 100
        status_messages.put(json.dumps({"progress": progress}))
    # handles fps
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(modules.globals.target_path, 30.0)
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    # clean and validate
    clean_temp(modules.globals.target_path)
    if is_video(modules.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')
    
    # Automatically open the generated output
    open_output(modules.globals.output_path)
    update_status(f'Output saved to {modules.globals.output_path}')

def update_status(message: str) -> None:
    logger.info(message)
    status_messages.put(json.dumps({"message": message}))

def open_output(output_path: str) -> None:
    if os.path.isfile(output_path):
        logger.info(f"Output file generated: {os.path.abspath(output_path)}")
    else:
        logger.error(f"Output file {output_path} does not exist.")

def destroy() -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    quit()

def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        logger.warning("GUI mode is not supported in this setup.")
        start()