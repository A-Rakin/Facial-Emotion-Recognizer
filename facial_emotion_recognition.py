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
