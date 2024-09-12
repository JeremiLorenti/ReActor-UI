import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps

import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, resolve_relative_path, save_last_used_paths, load_last_used_paths

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = 'Output'

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types

def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    # Load last used paths
    last_source_path, last_target_path = load_last_used_paths()
    if last_source_path and is_image(last_source_path):
        modules.globals.source_path = last_source_path
        image = render_image_preview(last_source_path, (200, 200))
        source_label.configure(image=image)
    if last_target_path and (is_image(last_target_path) or is_video(last_target_path)):
        modules.globals.target_path = last_target_path
        if is_image(last_target_path):
            image = render_image_preview(last_target_path, (200, 200))
            target_label.configure(image=image)
        elif is_video(last_target_path):
            video_frame = render_video_preview(last_target_path, (200, 200))
            target_label.configure(image=video_frame)

    return ROOT

def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    root = ctk.CTk()
    root.title(f"{modules.metadata.name} {modules.metadata.version}")
    root.geometry(f"{ROOT_WIDTH}x{ROOT_HEIGHT}")
    root.protocol("WM_DELETE_WINDOW", destroy)

    # Source file selection
    source_frame = ctk.CTkFrame(root)
    source_frame.pack(pady=10)
    source_button = ctk.CTkButton(source_frame, text="Select Source", command=lambda: select_file("source"))
    source_button.pack(side="left", padx=5)
    global source_label
    source_label = ctk.CTkLabel(source_frame, text="No source selected")
    source_label.pack(side="left", padx=5)

    # Target file selection
    target_frame = ctk.CTkFrame(root)
    target_frame.pack(pady=10)
    target_button = ctk.CTkButton(target_frame, text="Select Target", command=lambda: select_file("target"))
    target_button.pack(side="left", padx=5)
    global target_label
    target_label = ctk.CTkLabel(target_frame, text="No target selected")
    target_label.pack(side="left", padx=5)

    # Output directory selection
    output_frame = ctk.CTkFrame(root)
    output_frame.pack(pady=10)
    output_button = ctk.CTkButton(output_frame, text="Select Output Directory", command=select_output_directory)
    output_button.pack(side="left", padx=5)
    global status_label
    status_label = ctk.CTkLabel(output_frame, text=RECENT_DIRECTORY_OUTPUT)
    status_label.pack(side="left", padx=5)

    # Options
    options_frame = ctk.CTkFrame(root)
    options_frame.pack(pady=10)
    global face_enhancer_var, keep_fps_var, keep_audio_var, keep_frames_var, many_faces_var
    face_enhancer_var = ctk.BooleanVar()
    keep_fps_var = ctk.BooleanVar()
    keep_audio_var = ctk.BooleanVar(value=True)
    keep_frames_var = ctk.BooleanVar()
    many_faces_var = ctk.BooleanVar()
    ctk.CTkCheckBox(options_frame, text="Use Face Enhancer", variable=face_enhancer_var, command=toggle_face_enhancer)
    ctk.CTkCheckBox(options_frame, text="Keep FPS", variable=keep_fps_var)
    ctk.CTkCheckBox(options_frame, text="Keep Audio", variable=keep_audio_var)
    ctk.CTkCheckBox(options_frame, text="Keep Frames", variable=keep_frames_var)
    ctk.CTkCheckBox(options_frame, text="Many Faces", variable=many_faces_var)

    # Start button
    start_button = ctk.CTkButton(root, text="Start", command=start)
    start_button.pack(pady=10)

    return root

def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview

def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()

def toggle_face_enhancer() -> None:
    if face_enhancer_var.get():
        modules.globals.frame_processors.append('face_enhancer')
    else:
        modules.globals.frame_processors = [fp for fp in modules.globals.frame_processors if fp != 'face_enhancer']

def select_file(file_type: str) -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    if file_type == "source":
        path = ctk.filedialog.askopenfilename(title='Select an source image', initialdir=RECENT_DIRECTORY_SOURCE, filetypes=[img_ft])
        if is_image(path):
            modules.globals.source_path = path
            RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
            image = render_image_preview(modules.globals.source_path, (200, 200))
            source_label.configure(image=image)
            save_last_used_paths(modules.globals.source_path, modules.globals.target_path)  # Save paths
        else:
            modules.globals.source_path = None
            source_label.configure(image=None)
    elif file_type == "target":
        path = ctk.filedialog.askopenfilename(title='Select an target image or video', initialdir=RECENT_DIRECTORY_TARGET, filetypes=[img_ft, vid_ft])
        if is_image(path):
            modules.globals.target_path = path
            RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
            image = render_image_preview(modules.globals.target_path, (200, 200))
            target_label.configure(image=image)
            save_last_used_paths(modules.globals.source_path, modules.globals.target_path)  # Save paths
        elif is_video(path):
            modules.globals.target_path = path
            RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
            video_frame = render_video_preview(path, (200, 200))
            target_label.configure(image=video_frame)
            save_last_used_paths(modules.globals.source_path, modules.globals.target_path)  # Save paths
        else:
            modules.globals.target_path = None
            target_label.configure(image=None)

def select_output_directory() -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save image output file', filetypes=[img_ft], defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save video output file', filetypes=[vid_ft], defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        status_label.configure(text=modules.globals.output_path)

def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)

def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()

def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()

def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)

def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw == False:
            from modules.predicter import predict_frame
            if predict_frame(temp_frame):
                quit()
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)),
                temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

def display_output(output_path: str) -> None:
    if is_image(output_path):
        image = render_image_preview(output_path, (400, 400))  # Adjust size as needed
        preview_label.configure(image=image)
        preview_label.image = image  # Keep a reference to avoid garbage collection
    elif is_video(output_path):
        video_frame = render_video_preview(output_path, (400, 400))  # Adjust size as needed
        preview_label.configure(image=video_frame)
        preview_label.image = video_frame  # Keep a reference to avoid garbage collection
    PREVIEW.deiconify()
