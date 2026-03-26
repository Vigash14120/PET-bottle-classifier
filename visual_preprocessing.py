import cv2
import numpy as np

def preprocess_image(frame, target_size=(150, 150)):
    """
    Implements Visual Preprocessing as per the workflow diagram:
    1. Resize (150x150)
    2. Gaussian Denoising
    3. Normalization (0-1)
    """
    if frame is None:
        return None

    # 1. Resize
    resized = cv2.resize(frame, target_size)

    # 2. Gaussian Denoising (using a 5x5 kernel as standard)
    denoised = cv2.GaussianBlur(resized, (5, 5), 0)

    # 3. Normalization (Scale 0-255 pixels to 0.0-1.0 range)
    normalized = denoised.astype(np.float32) / 255.0

    return normalized

if __name__ == "__main__":
    # Test block
    dummy_frame = np.uint8(np.random.rand(480, 640, 3) * 255)
    processed = preprocess_image(dummy_frame)
    print(f"✅ Visual Preprocessing Test: Shape {processed.shape}, Max {np.max(processed):.2f}")
