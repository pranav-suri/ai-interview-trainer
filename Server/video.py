import tensorflow as tf
from tensorflow.keras.optimizers import Adam # Add optimizer import (needed for compile)
import mediapipe as mp # For face mesh and pose estimation
import cv2
import numpy as np
import time

# --- Define Model Architecture ---
IMG_SIZE = (48, 48) # Make sure this matches training
NUM_CLASSES = 7     # Make sure this matches training

def create_model_simple():
    model = tf.keras.models.Sequential([
        # NOTE: Define input_shape *without* the batch dimension here
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # Use NUM_CLASSES
    ])
    # Compile is necessary after loading weights for the model to be usable for prediction
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Model Loading and Configuration ---
model_path = 'best_model_full.h5'  # Make sure this path is correct
loaded_emotion_model = None # Initialize as None

try:
    # 1. Create the model architecture
    loaded_emotion_model = create_model_simple()
    # 2. Load only the weights
    loaded_emotion_model.load_weights(model_path)
    print(f"Custom emotion model architecture created and weights loaded successfully from {model_path}")
    # loaded_emotion_model.summary() # Optional: check summary
except Exception as e:
    print(f"Error creating model architecture or loading weights: {e}")
    loaded_emotion_model = None # Ensure it's None if loading fails

# --- Define Class Labels (Ensure this order matches training) ---
# Based on alphabetical sorting typically used by ImageDataGenerator:
class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
target_size = (48, 48) # The input size your model expects

# --- Constants and MediaPipe Initialization (Keep as is) ---
VISUALIZE = True
WINDOW_NAME = 'Interview Analysis Visualization'
VIDEO_ANALYSIS_FRAME_SKIP = 2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True, min_tracking_confidence=0.5)
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_video_features(video_path):
    """
    Analyzes video for facial expressions (using custom Keras model), eye contact, posture,
    and provides visual feedback if VISUALIZE is True.
    """
    if loaded_emotion_model is None:
        print("Emotion model not loaded. Skipping emotion analysis.")
        # Handle how you want the function to behave if the model isn't loaded
        # Maybe return partial results or None

    print("Analyzing video features (using custom model)...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    frame_count = 0
    processed_frame_count = 0
    # Initialize emotion counts dictionary using the defined labels
    emotion_counts = {label: 0 for label in class_labels}
    eye_contact_frames = 0
    upright_frames = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if VISUALIZE:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        if frame_count % VIDEO_ANALYSIS_FRAME_SKIP != 0:
            continue # Skip frame

        processed_frame_count += 1
        start_time_frame = time.time()

        annotated_frame = frame.copy()
        rgb_frame_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Use RGB for MediaPipe
        rgb_frame_mp.flags.writeable = False

        # --- MediaPipe Face Mesh and Pose Processing ---
        face_results = face_mesh.process(rgb_frame_mp)
        pose_results = pose_estimator.process(rgb_frame_mp)

        rgb_frame_mp.flags.writeable = True # Re-enable if needed later

        # --- Facial Emotion Analysis (Using Custom Keras Model) ---
        current_emotion = "N/A" # Default value

        if loaded_emotion_model: # Check if the model was loaded successfully
            try:
                # 1. Preprocess the *current frame* for the emotion model
                # Convert frame to RGB (if not already done for MP)
                rgb_frame_emotion = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to the target size (48x48)
                img_resized = cv2.resize(rgb_frame_emotion, target_size)
                # Convert to float and rescale pixel values to [0, 1]
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                # Add the batch dimension (shape becomes 1, 48, 48, 3)
                img_batched = np.expand_dims(img_array, axis=0)

                # 2. Predict using the loaded Keras model
                # Use verbose=0 to avoid printing progress for every frame
                predictions = loaded_emotion_model.predict(img_batched, verbose=0)

                # 3. Interpret the prediction
                predicted_index = np.argmax(predictions[0]) # Get index of max probability
                current_emotion = class_labels[predicted_index] # Map index to label

                # Update counts
                emotion_counts[current_emotion] += 1

            except Exception as e:
                # Catch potential errors during preprocessing or prediction
                # print(f"Frame {frame_count}: Custom Emotion Model error: {e}") # Optional for debugging
                current_emotion = "Error" # Indicate an error occurred for this frame
                pass # Continue processing the video
        is_eye_contact = False # Flag for current frame
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh
                if VISUALIZE:
                    mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    # Iris landmarks drawing (optional, can be busy)
                    # mp_drawing.draw_landmarks(
                    #     image=annotated_frame,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                # Calculate eye contact
                try:
                    left_pupil = face_landmarks.landmark[473] # Corrected index? Check docs if unsure. Typically 473-477 are right iris, 468-472 left. Let's use documented iris centers.
                    right_pupil = face_landmarks.landmark[468] # Corrected index? Let's use documented iris centers.

                    # Get corners for width calculation
                    left_eye_inner = face_landmarks.landmark[133]
                    left_eye_outer = face_landmarks.landmark[33]
                    right_eye_inner = face_landmarks.landmark[362]
                    right_eye_outer = face_landmarks.landmark[263]

                    left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
                    right_eye_width = abs(right_eye_outer.x - right_eye_inner.x)

                    if left_eye_width > 0.01 and right_eye_width > 0.01: # Avoid division by zero
                        # Use the correct pupil landmark index for relative position calculation
                        left_pupil_rel_pos = (face_landmarks.landmark[468].x - left_eye_inner.x) / left_eye_width # Use left pupil center [468]
                        right_pupil_rel_pos = (face_landmarks.landmark[473].x - right_eye_inner.x) / right_eye_width # Use right pupil center [473]

                        # Thresholds for looking 'forward' (TUNING NEEDED!)
                        if 0.3 < left_pupil_rel_pos < 0.7 and 0.3 < right_pupil_rel_pos < 0.7:
                            is_eye_contact = True
                            eye_contact_frames += 1
                            # print("Eye contact DETECTED") # Debug print

                    # Visualize Eye Contact state (draw pupils differently)
                    if VISUALIZE:
                        pupil_color = (0, 255, 0) if is_eye_contact else (0, 0, 255) # Green if contact, Red if not
                        # Get pixel coordinates
                        l_pupil_px = mp_drawing._normalized_to_pixel_coordinates(face_landmarks.landmark[468].x, face_landmarks.landmark[468].y, frame_width, frame_height)
                        r_pupil_px = mp_drawing._normalized_to_pixel_coordinates(face_landmarks.landmark[473].x, face_landmarks.landmark[473].y, frame_width, frame_height)
                        if l_pupil_px and r_pupil_px:
                            cv2.circle(annotated_frame, l_pupil_px, 3, pupil_color, -1)
                            cv2.circle(annotated_frame, r_pupil_px, 3, pupil_color, -1)

                    break # Process only the first detected face

                except IndexError:
                     print(f"Warning: Iris landmarks (468, 473) not found. Ensure 'refine_landmarks=True' is set.")
                     # Draw basic mesh even if iris fails
                     if VISUALIZE:
                         mp_drawing.draw_landmarks(
                            image=annotated_frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                         mp_drawing.draw_landmarks(
                            image=annotated_frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                except Exception as e_eye:
                    print(f"Error during eye contact calculation/drawing: {e_eye}")


        # --- Posture Heuristic Calculation & Visualization ---
        is_upright = False # Flag for current frame
        if pose_results.pose_landmarks:
             # Draw the pose skeleton
            if VISUALIZE:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Calculate posture
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            if left_shoulder.visibility > 0.6 and right_shoulder.visibility > 0.6:
                y_diff = abs(left_shoulder.y - right_shoulder.y) * frame_height # Pixel diff
                if y_diff < frame_height * 0.1: # Shoulders relatively level
                    is_upright = True
                    upright_frames += 1

            # Visualize Posture state (draw shoulder line differently)
            if VISUALIZE:
                posture_color = (0, 255, 0) if is_upright else (0, 0, 255) # Green if upright, Red if not
                ls_px = mp_drawing._normalized_to_pixel_coordinates(left_shoulder.x, left_shoulder.y, frame_width, frame_height)
                rs_px = mp_drawing._normalized_to_pixel_coordinates(right_shoulder.x, right_shoulder.y, frame_width, frame_height)
                if ls_px and rs_px and left_shoulder.visibility > 0.6 and right_shoulder.visibility > 0.6:
                    cv2.line(annotated_frame, ls_px, rs_px, posture_color, 2)


        # --- Display Text Info on Frame ---
        if VISUALIZE:
            y_pos = 30 # Starting Y position for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_color = (255, 255, 255) # White
            bg_color = (0, 0, 0) # Black background for text
            thickness = 2

            # Emotion Text
            text_emotion = f"Emotion: {current_emotion}"
            (w, h), _ = cv2.getTextSize(text_emotion, font, font_scale, thickness)
            cv2.rectangle(annotated_frame, (10, y_pos - h - 5) , (10 + w + 5, y_pos + 5), bg_color, -1)
            cv2.putText(annotated_frame, text_emotion, (10, y_pos), font, font_scale, text_color, thickness)
            y_pos += h + 10

            # Eye Contact Text
            text_eye = f"Eye Contact: {'YES' if is_eye_contact else 'NO'}"
            (w, h), _ = cv2.getTextSize(text_eye, font, font_scale, thickness)
            cv2.rectangle(annotated_frame, (10, y_pos - h - 5) , (10 + w + 5, y_pos + 5), bg_color, -1)
            cv2.putText(annotated_frame, text_eye, (10, y_pos), font, font_scale, (0, 255, 0) if is_eye_contact else (0, 0, 255), thickness)
            y_pos += h + 10

            # Posture Text
            text_posture = f"Posture: {'Upright' if is_upright else 'Not Upright'}"
            (w, h), _ = cv2.getTextSize(text_posture, font, font_scale, thickness)
            cv2.rectangle(annotated_frame, (10, y_pos - h - 5) , (10 + w + 5, y_pos + 5), bg_color, -1)
            cv2.putText(annotated_frame, text_posture, (10, y_pos), font, font_scale, (0, 255, 0) if is_upright else (0, 0, 255), thickness)

        # --- Display Frame ---
        if VISUALIZE:
            end_time_frame = time.time()
            processing_time = end_time_frame - start_time_frame
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (frame_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow(WINDOW_NAME, annotated_frame)
            # Allow window events and check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Visualization stopped by user ('q' pressed).")
                break

        # Progress print to console
        if processed_frame_count % 20 == 0: # Print progress less often now
             print(f"...processed video frame {frame_count} (Total processed: {processed_frame_count})")

    # --- Cleanup and Final Calculations ---
    cap.release()
    if VISUALIZE:
        cv2.destroyAllWindows()
    print("Video analysis complete.")

    # Compile results (same as before)
    video_analysis_results = {}
    if processed_frame_count > 0:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "N/A"
        video_analysis_results['dominant_emotion'] = dominant_emotion
        video_analysis_results['emotion_distribution'] = emotion_counts
        # video_analysis_results['eye_contact_percentage'] = round((eye_contact_frames / processed_frame_count) * 100, 2)
        video_analysis_results['upright_posture_percentage'] = round((upright_frames / processed_frame_count) * 100, 2)

        print(f"- Dominant Emotion Detected: {dominant_emotion}")
        print(f"- Emotion Distribution: {emotion_counts}")
        # print(f"- Estimated Eye Contact: {video_analysis_results['eye_contact_percentage']}%")
        print(f"- Estimated Upright Posture: {video_analysis_results['upright_posture_percentage']}%")
    else:
        print("- No frames processed for video analysis.")
        video_analysis_results['error'] = "No frames processed"

    return video_analysis_results