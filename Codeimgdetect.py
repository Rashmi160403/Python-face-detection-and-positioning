import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video capture (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterate through the detected faces and apply blur
    for (x, y, w, h) in faces:
        # Extract the region of interest (the detected face)
        face_roi = frame[y:y+h, x:x+w]

        # Apply Gaussian blur to the detected face
        face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)

        # Replace the original face with the blurred face
        frame[y:y+h, x:x+w] = face_roi

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the position of the detected face
        cv2.putText(frame, f'Face Position: ({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected and blurred faces
    cv2.imshow('Face Detection and Blur', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
