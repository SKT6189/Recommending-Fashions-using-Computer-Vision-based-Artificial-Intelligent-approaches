import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to process the input image and apply face mesh detection
def detect_face_mesh(input_image_path, output_image_path):
    # Read the input image
    input_image = cv2.imread(input_image_path)

    # Convert the image to RGB format
    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Detect face mesh
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        # Iterate over detected faces
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks on the image
            for landmark in face_landmarks.landmark:
                x, y, z = int(landmark.x * input_image.shape[1]), int(landmark.y * input_image.shape[0]), landmark.z
                cv2.circle(input_image, (x, y), 2, (0, 255, 0), -1)

    # Save the output image
    cv2.imwrite(output_image_path, input_image)

    # Release resources
    face_mesh.close()

if __name__ == "__main__":
    input_image_path = "./Screenshot_Sahil.jpg"  # Replace with the path to your input image
    output_image_path = "./output.jpg"  # Replace with the desired output image path

    detect_face_mesh(input_image_path, output_image_path)
    print(f"Face mesh detection completed. Result saved at {output_image_path}")
