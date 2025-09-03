Opens webcam (cv2.VideoCapture(0)).

Reads each frame in a loop.

Converts the image to RGB (because FER expects it).

Detects faces + emotions in that frame.

For each face:

Finds the dominant emotion (highest probability).

Draws a rectangle around the face.

Shows the emotion + confidence score above the face.

Updates in real-time until you press q.
