import cv2
import sys

def test_camera():
    for i in range(5):
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Success! Camera index {i} is working.")
                cap.release()
                return
            else:
                print(f"Camera index {i} opened but failed to read frame.")
        else:
            print(f"Camera index {i} could not be opened.")
        cap.release()
    print("No working camera found.")

if __name__ == "__main__":
    test_camera()
