"""
Python script to take a video file as input and extract each frame into a directory of images
"""
import os
import sys
import cv2

def extract_frames(path: str, modulo=1):
    video_dir = os.path.join(os.getcwd(),path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Could not open {path}")
        return
    
    count_written = 0
    frame_count = 0
    while True:
        ret,frame = cap.get()
        if not ret:
            print(f"Wrote {frame_count} frames")
            return
        if frame_count % modulo != 0:
            frame_count += 1
            return
        
        frame_count += 1
        cv2.imwrite(os.path.join(video_dir,f"{frame_count}"),frame)
        count_written += 1

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Must pass in a video as command line argument")
        sys.exit()
    if len(sys.argv < 3):
        modulo = 1
    else:
        modulo = int(sys.argv[2])
    print(f"Extracting one frame for every {modulo} frame of {sys.argv[1]}")
    extract_frames(path=sys.argv[1],modulo=1)