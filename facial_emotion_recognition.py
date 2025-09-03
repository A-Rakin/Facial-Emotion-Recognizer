import cv2
from fer import FER
# Initialize the emotion detector
emotion_detector = FER()

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV's BGR frame to RGB (FER expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect emotions in the frame
    emotions = emotion_detector.detect_emotions(rgb_frame)

    # Debug: print detected emotions to console
    print("Detected emotions:", emotions)

    if emotions:
        # Get the first detected face's emotions
        emotion_data = emotions[0]['emotions']

        # Find the emotion with the highest confidence
        max_emotion = max(emotion_data, key=emotion_data.get)

        # Get the coordinates for the face
        (x, y, w, h) = emotions[0]["box"]

        # Display the emotion on the frame
        cv2.putText(frame, max_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


"""

Step-by-Step Breakdown

Import libraries

cv2 → for video and image processing.

FER → pre-trained Facial Emotion Recognition model.

Initialize the detector

emotion_detector = FER()


This creates the object that can recognize emotions from images.

Start the webcam

cap = cv2.VideoCapture(0)


The 0 tells OpenCV to use your default webcam.

Read frames in a loop

ret, frame = cap.read()


ret is True if a frame is captured successfully.

frame is the image captured from your webcam.

Convert color format

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


OpenCV uses BGR, but the FER model expects RGB.

Detect emotions

emotions = emotion_detector.detect_emotions(rgb_frame)


This gives you a list of faces found, with their:

Bounding box (x, y, w, h) → where the face is.

Emotion probabilities (happy: 0.75, sad: 0.1, etc.).

Pick the strongest emotion

emotion_data = emotions[0]['emotions']
max_emotion = max(emotion_data, key=emotion_data.get)


emotions[0] → looks at the first detected face.

max_emotion → finds which emotion has the highest confidence score.

Draw results on screen

cv2.rectangle → draws a blue box around the face.

cv2.putText → writes the detected emotion above the face.

Show live video

cv2.imshow('Emotion Detector', frame)


Opens a window displaying the webcam with emotions written.

Exit condition

if cv2.waitKey(1) & 0xFF == ord('q'):
    break


If you press q, the program stops.

Release resources

cap.release()
cv2.destroyAllWindows()


Closes the webcam and the display window properly.

✅ In simple terms:

The webcam captures frames continuously.

Each frame is analyzed by the FER model.

The model finds faces, checks which emotion is most likely, and displays it on the screen with a box.

Keeps running until you press q.

"""