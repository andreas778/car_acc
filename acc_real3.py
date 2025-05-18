import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Specific to MobileNetV2
import time
import os
import datetime
from collections import deque # For frame buffer
import subprocess # For running ffmpeg

# --- Configuration ---
MODEL_JSON_PATH = "transfer_model_v1.json" # Your trained model
MODEL_WEIGHTS_PATH = "transfer_model_v1.weights.h5" # Your trained weights
INPUT_IMG_SIZE = (224, 224)

# !!! IMPORTANT: Set this based on your training script output !!!
ACCIDENT_CLASS_INDEX = 1 # Example: 1 if 'Accident' is the second class alphabetically
ACCIDENT_THRESHOLD = 0.6
MIN_BRIGHTNESS_THRESHOLD = 10.0

# --- Video Clip Saving Configuration ---
SAVE_CLIPS_DEFAULT = True
CLIPS_OUTPUT_DIR = "accident_clips" # Directory to save clips
CLIP_PRE_ROLL_SECS = 3
CLIP_POST_ROLL_SECS = 4
FRAMES_TO_TRIGGER_ACCIDENT = 3
FRAMES_TO_END_ACCIDENT = 10 # Consecutive non-accident frames to end event

# --- FFmpeg Configuration ---
USE_FFMPEG_CONVERSION = True # Set to False to disable ffmpeg post-processing
FFMPEG_PATH = "ffmpeg" # Assumes ffmpeg is in PATH, otherwise provide full path e.g., "/usr/bin/ffmpeg"
# Standard ffmpeg options for web-compatible H.264 MP4
FFMPEG_OUTPUT_ARGS = [
    '-c:v', 'libx264',
    '-preset', 'medium', # 'ultrafast' for speed, 'medium' for balance
    '-crf', '23',
    '-pix_fmt', 'yuv420p',
    '-c:a', 'aac',
    '-strict', 'experimental', # For AAC in some ffmpeg versions
    '-movflags', '+faststart'
]

# --- State Definitions ---
STATE_NORMAL = 0
STATE_ACCIDENT_DETECTED = 1
STATE_POST_ROLL = 2
# -----------------------------------------------------

# --- Helper Functions ---
@st.cache_resource
def load_accident_model(json_path, weights_path):
    if not os.path.exists(json_path) or not os.path.exists(weights_path):
         st.error(f"Error: Model files not found. Ensure '{json_path}' and '{weights_path}' are present.")
         return None
    try:
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        #st.success(f"Model loaded successfully from {json_path} and {weights_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_frame(frame, target_size):
    img_resized = cv2.resize(frame, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32)
    img_batch = np.expand_dims(img_float, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    return img_preprocessed

def create_video_writer(output_dir, frame_width, frame_height, fps, temp_suffix="_temp_opencv"):
    if not os.path.exists(output_dir):
        try: os.makedirs(output_dir)
        except OSError as e: st.error(f"Error creating directory {output_dir}: {e}"); return None, None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # Save initial OpenCV output with a temporary suffix if FFmpeg is enabled
    filename_base = f"accident_{timestamp}"
    if USE_FFMPEG_CONVERSION:
        filename = os.path.join(output_dir, f"{filename_base}{temp_suffix}.mp4")
    else:
        filename = os.path.join(output_dir, f"{filename_base}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        st.error(f"Error: Could not open video writer for {filename}")
        return None, None
    print(f"Started recording initial clip: {filename}")
    return writer, filename, filename_base # Return base for ffmpeg naming

def convert_video_with_ffmpeg(input_path, output_base_filename, output_dir):
    """Converts video, returns path to converted file or original if failed."""
    global USE_FFMPEG_CONVERSION # To disable permanently if ffmpeg not found
    if not os.path.exists(input_path):
        st.error(f"FFmpeg input file not found: {input_path}")
        return input_path # Fallback to original

    output_path = os.path.join(output_dir, f"{output_base_filename}_web.mp4")

    try:
        command = [FFMPEG_PATH, '-y', '-i', input_path] + FFMPEG_OUTPUT_ARGS + [output_path]
        st.toast(f"Processing clip with FFmpeg: {os.path.basename(output_path)}", icon="⚙️")
        print(f"Running FFmpeg: {' '.join(command)}")
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"FFmpeg conversion successful: {input_path} -> {output_path}")
        try:
            os.remove(input_path) # Remove temp opencv file
            print(f"Removed temporary OpenCV file: {input_path}")
        except OSError as e:
            print(f"Warning: Could not remove temp file {input_path}: {e}")
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg error for {os.path.basename(input_path)}: {e.stderr[:500]}...") # Show first 500 chars of error
        print(f"FFmpeg stderr for {input_path}: {e.stderr}")
        return input_path # Fallback to original on error
    except FileNotFoundError:
        st.error(f"FFmpeg not found at '{FFMPEG_PATH}'. Disabling FFmpeg conversion.")
        USE_FFMPEG_CONVERSION = False # Disable for the rest of the session
        return input_path # Fallback


# --- Main Application ---
st.set_page_config(layout="wide", page_title="Real-time Accident Detection")
st.title("Car Accident Detection System")
#st.caption(f"Using model: {MODEL_JSON_PATH}")

if 'saved_clips_session' not in st.session_state: st.session_state.saved_clips_session = []
if 'accident_state_session' not in st.session_state: st.session_state.accident_state_session = STATE_NORMAL
if 'current_clip_writer_session' not in st.session_state: st.session_state.current_clip_writer_session = None
if 'current_clip_filename_session' not in st.session_state: st.session_state.current_clip_filename_session = None
if 'current_clip_basename_session' not in st.session_state: st.session_state.current_clip_basename_session = None


model = load_accident_model(MODEL_JSON_PATH, MODEL_WEIGHTS_PATH)

if model:
    st.sidebar.header("Controls")
    run_detection_checkbox = st.sidebar.checkbox("Start Camera Feed & Detection", value=False)
    confidence_threshold_slider = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, ACCIDENT_THRESHOLD, 0.05)
    save_clips_option_checkbox = st.sidebar.checkbox("Save Accident Clips", value=SAVE_CLIPS_DEFAULT)
    #use_ffmpeg_checkbox = st.sidebar.checkbox("Use FFmpeg for Web Compatibility", value=USE_FFMPEG_CONVERSION)


    video_placeholder = st.empty()
    st.markdown("---")
    clips_display_area = st.container()

    if not run_detection_checkbox:
        st.session_state.accident_state_session = STATE_NORMAL
        if st.session_state.current_clip_writer_session:
             print("Releasing writer due to detection stop (checkbox).")
             st.session_state.current_clip_writer_session.release()
             # FFmpeg conversion for abruptly stopped clip
             opencv_clip = st.session_state.current_clip_filename_session
             base_clip_name = st.session_state.current_clip_basename_session
             finalized_clip = opencv_clip
             if opencv_clip and os.path.exists(opencv_clip):
                 finalized_clip = convert_video_with_ffmpeg(opencv_clip, base_clip_name, CLIPS_OUTPUT_DIR)

             if finalized_clip and os.path.exists(finalized_clip):
                 if finalized_clip not in st.session_state.saved_clips_session:
                    st.session_state.saved_clips_session.insert(0, finalized_clip)
             st.session_state.current_clip_writer_session = None
             st.session_state.current_clip_filename_session = None
             st.session_state.current_clip_basename_session = None


    if run_detection_checkbox:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): video_placeholder.error("Error: Could not open webcam.")
        else:
            video_placeholder.info("Webcam opened. Starting detection...")
            webcam_fps = cap.get(cv2.CAP_PROP_FPS);
            if webcam_fps <= 0 or webcam_fps > 120: webcam_fps = 20.0; st.toast(f"Estimating FPS: {webcam_fps:.1f}", icon="⚠️")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            buffer_size_frames = int(CLIP_PRE_ROLL_SECS * webcam_fps); frame_buffer = deque(maxlen=buffer_size_frames)

            fps_start_time = time.time(); fps_frame_count = 0; fps_text = "FPS: --.--"
            consecutive_accident_frames = 0; consecutive_no_accident_frames = 0; post_roll_end_time = None

            if 'run_detection_active' not in st.session_state: st.session_state.run_detection_active = True
            else: st.session_state.run_detection_active = True # Reset if previously false

            while st.session_state.run_detection_active:
                if not run_detection_checkbox: st.session_state.run_detection_active = False; break
                ret, frame = cap.read(); current_time = time.time()
                if not ret: st.toast("Can't grab frame.", icon="❌"); st.session_state.run_detection_active = False; break
                frame_buffer.append((current_time, frame.copy())); display_frame = frame.copy()

                label = "No Accident (Unknown)"; color = (0,255,0); is_currently_accident = False
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); avg_brightness = np.mean(gray_frame)
                if avg_brightness < MIN_BRIGHTNESS_THRESHOLD:
                    label = f"Too Dark ({avg_brightness:.1f})"; color = (80,80,80); consecutive_accident_frames = 0; consecutive_no_accident_frames += 1
                else:
                    processed_batch = preprocess_frame(frame, INPUT_IMG_SIZE)
                    try:
                        prediction = model.predict(processed_batch, verbose=0)[0]
                        accident_prob = prediction[ACCIDENT_CLASS_INDEX]
                        if accident_prob >= confidence_threshold_slider:
                            is_currently_accident = True; consecutive_accident_frames += 1; consecutive_no_accident_frames = 0
                            label = f"Accident ({accident_prob:.2f})"; color = (0,0,255)
                        else:
                            is_currently_accident = False; consecutive_no_accident_frames += 1; consecutive_accident_frames = 0
                            no_accident_prob = prediction[1 - ACCIDENT_CLASS_INDEX]; label = f"No Accident ({no_accident_prob:.2f})"; color = (0,255,0)
                    except Exception as e:
                        st.toast(f"Prediction error: {e}", icon="🔥"); label = "Pred Error"; color = (0,255,255)
                        consecutive_accident_frames = 0; consecutive_no_accident_frames +=1

                if save_clips_option_checkbox:
                    current_accident_state = st.session_state.accident_state_session
                    if current_accident_state == STATE_NORMAL:
                        if consecutive_accident_frames >= FRAMES_TO_TRIGGER_ACCIDENT:
                            st.session_state.accident_state_session = STATE_ACCIDENT_DETECTED
                            st.toast("Accident Event! Recording...", icon="🔴")
                            writer, filename, basename = create_video_writer(CLIPS_OUTPUT_DIR, frame_width, frame_height, webcam_fps)
                            st.session_state.current_clip_writer_session = writer
                            st.session_state.current_clip_filename_session = filename # This is the temp OpenCV filename
                            st.session_state.current_clip_basename_session = basename # Base for FFmpeg naming
                            if writer:
                                for _, buffered_frame in list(frame_buffer): writer.write(buffered_frame)
                                writer.write(frame)
                            else: st.session_state.accident_state_session = STATE_NORMAL
                    elif current_accident_state == STATE_ACCIDENT_DETECTED:
                        writer = st.session_state.current_clip_writer_session
                        if writer:
                            writer.write(frame)
                            if consecutive_no_accident_frames >= FRAMES_TO_END_ACCIDENT:
                                if CLIP_POST_ROLL_SECS > 0:
                                    st.session_state.accident_state_session = STATE_POST_ROLL
                                    post_roll_end_time = current_time + CLIP_POST_ROLL_SECS
                                    print(f"Post-roll for {st.session_state.current_clip_basename_session}...")
                                else: # No post-roll
                                    print(f"Finalizing (no post-roll): {st.session_state.current_clip_basename_session}")
                                    writer.release()
                                    opencv_clip = st.session_state.current_clip_filename_session
                                    base_clip_name = st.session_state.current_clip_basename_session
                                    final_clip = opencv_clip
                                    if opencv_clip and os.path.exists(opencv_clip):
                                        final_clip = convert_video_with_ffmpeg(opencv_clip, base_clip_name, CLIPS_OUTPUT_DIR)
                                    if final_clip and os.path.exists(final_clip):
                                        if final_clip not in st.session_state.saved_clips_session:
                                            st.session_state.saved_clips_session.insert(0, final_clip)
                                    st.session_state.current_clip_writer_session = None; st.session_state.current_clip_filename_session = None; st.session_state.current_clip_basename_session = None
                                    st.session_state.accident_state_session = STATE_NORMAL
                                    consecutive_accident_frames = 0; consecutive_no_accident_frames = 0
                        else: st.session_state.accident_state_session = STATE_NORMAL
                    elif current_accident_state == STATE_POST_ROLL:
                        writer = st.session_state.current_clip_writer_session
                        if writer:
                            writer.write(frame)
                            if post_roll_end_time is None or current_time >= post_roll_end_time:
                                print(f"Post-roll finished. Finalizing: {st.session_state.current_clip_basename_session}")
                                writer.release()
                                opencv_clip = st.session_state.current_clip_filename_session
                                base_clip_name = st.session_state.current_clip_basename_session
                                final_clip = opencv_clip
                                if opencv_clip and os.path.exists(opencv_clip):
                                    final_clip = convert_video_with_ffmpeg(opencv_clip, base_clip_name, CLIPS_OUTPUT_DIR)
                                if final_clip and os.path.exists(final_clip):
                                    if final_clip not in st.session_state.saved_clips_session:
                                        st.session_state.saved_clips_session.insert(0, final_clip)
                                st.session_state.current_clip_writer_session = None; st.session_state.current_clip_filename_session = None; st.session_state.current_clip_basename_session = None
                                st.session_state.accident_state_session = STATE_NORMAL
                                consecutive_accident_frames = 0; consecutive_no_accident_frames = 0; post_roll_end_time = None
                        else: st.session_state.accident_state_session = STATE_NORMAL

                fps_frame_count += 1; elapsed_time = time.time() - fps_start_time
                if elapsed_time >= 1.0: fps = fps_frame_count / elapsed_time; fps_text = f"FPS: {fps:.2f}"; fps_start_time = time.time(); fps_frame_count = 0
                status_text = ""; status_color = (255,255,255)
                if st.session_state.accident_state_session == STATE_ACCIDENT_DETECTED: status_text = "REC (Event)"; status_color = (0,0,255)
                elif st.session_state.accident_state_session == STATE_POST_ROLL: status_text = "REC (Post-Roll)"; status_color = (0,165,255)
                cv2.putText(display_frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display_frame, fps_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                if status_text: cv2.putText(display_frame, status_text, (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                video_placeholder.image(display_frame, channels="BGR", use_container_width=True)

                if st.session_state.accident_state_session == STATE_NORMAL and ('last_displayed_clip_count' not in st.session_state or st.session_state.last_displayed_clip_count != len(st.session_state.saved_clips_session)):
                    with clips_display_area:
                        clips_display_area.empty()
                        if st.session_state.saved_clips_session:
                            st.subheader("Detected Accident Clips:")
                            for clip_path in st.session_state.saved_clips_session:
                                try: st.video(clip_path); st.caption(os.path.basename(clip_path))
                                except Exception as e: st.error(f"Display err {os.path.basename(clip_path)}: {e}")
                    st.session_state.last_displayed_clip_count = len(st.session_state.saved_clips_session)

            cap.release()
            if st.session_state.current_clip_writer_session:
                print(f"Loop ended. Finalizing: {st.session_state.current_clip_basename_session}")
                st.session_state.current_clip_writer_session.release()
                opencv_clip = st.session_state.current_clip_filename_session
                base_clip_name = st.session_state.current_clip_basename_session
                final_clip = opencv_clip
                if opencv_clip and os.path.exists(opencv_clip):
                    final_clip = convert_video_with_ffmpeg(opencv_clip, base_clip_name, CLIPS_OUTPUT_DIR)
                if final_clip and os.path.exists(final_clip):
                    if final_clip not in st.session_state.saved_clips_session:
                        st.session_state.saved_clips_session.insert(0, final_clip)
                st.session_state.current_clip_writer_session = None; st.session_state.current_clip_filename_session = None; st.session_state.current_clip_basename_session = None
            video_placeholder.info("Camera feed stopped. Check sidebar to restart.")
            st.session_state.run_detection_active = False

    if not run_detection_checkbox:
        with clips_display_area:
            clips_display_area.empty()
            if st.session_state.saved_clips_session:
                st.subheader("Saved Accident Clips (Current Session):")
                for clip_path in st.session_state.saved_clips_session:
                    try: st.video(clip_path); st.caption(os.path.basename(clip_path))
                    except Exception as e: st.error(f"Display err {os.path.basename(clip_path)}: {e}")
            else: st.info("Start detection to record clips.")
        if 'last_displayed_clip_count' in st.session_state: del st.session_state.last_displayed_clip_count
else:
    st.warning("Model could not be loaded.")
st.caption(f"App Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
