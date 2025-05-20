import cv2
import numpy as np
import csv
from collections import deque
from scipy.spatial.distance import pdist, squareform

# Parameters (adjust these for your video)
MIN_CROWD_SIZE = 3            # Minimum persons in a crowd
DISTANCE_THRESHOLD = 150      # Max distance (in pixels) to consider people "close"
PERSISTENCE_FRAMES = 3        # Number of consecutive frames crowd must persist (lowered for debugging)

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Video path
video_path = 'dataset_video.mp4'

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_count = 0

# To keep track of persistent crowds
persistent_crowds = deque(maxlen=PERSISTENCE_FRAMES)
crowd_log = []  # To store detected crowd events: (frame_number, crowd_size)

def find_crowds(centers, min_size=MIN_CROWD_SIZE, max_dist=DISTANCE_THRESHOLD):
    if len(centers) < min_size:
        return []
    dists = squareform(pdist(centers))
    groups = []
    visited = set()

    for i in range(len(centers)):
        if i in visited:
            continue
        group = {i}
        # BFS to find connected cluster within distance threshold
        queue = [i]
        while queue:
            idx = queue.pop()
            for j in range(len(centers)):
                if j != idx and j not in group and dists[idx][j] < max_dist:
                    group.add(j)
                    queue.append(j)
        if len(group) >= min_size:
            groups.append(group)
        visited.update(group)
    return groups

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detect people
    rects, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)

    centers = []
    for (x, y, w, h) in rects:
        cx = x + w//2
        cy = y + h//2
        centers.append((cx, cy))

    print(f"Frame {frame_count}: Detected {len(centers)} persons")

    # Find crowds in this frame
    current_crowds = find_crowds(centers)

    # For debugging: print crowd groups
    for group in current_crowds:
        print(f"Frame {frame_count}: Crowd group with {len(group)} persons")

    # Save current crowds to persistent queue
    persistent_crowds.append(current_crowds)

    # Check persistence of crowds over last frames
    if len(persistent_crowds) == PERSISTENCE_FRAMES:
        # For each crowd group in current frame
        for crowd in persistent_crowds[-1]:
            # Check if similar group existed in previous frames
            persistence_count = 1
            for past_crowds in list(persistent_crowds)[:-1]:
                # Check overlap with any group in past frame
                if any(len(crowd.intersection(past_group)) >= MIN_CROWD_SIZE // 2 for past_group in past_crowds):
                    persistence_count += 1
                else:
                    break
            if persistence_count == PERSISTENCE_FRAMES:
                # Log crowd detection
                crowd_log.append((frame_count, len(crowd)))
                print(f"CROWD DETECTED: Frame {frame_count}, Persons: {len(crowd)}")

    # Show frame with detections (optional)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow('Crowd Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit early
        break

cap.release()
cv2.destroyAllWindows()

# Write results to CSV
with open('crowd_detection_log.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Frame Number', 'Person Count in Crowd'])
    for row in crowd_log:
        csvwriter.writerow(row)

print("Detection complete. Results saved to crowd_detection_log.csv")
