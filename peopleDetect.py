import cv2
import sys
import os
import shutil


def to_frames(video):
  os.makedirs("frames")
  vidcap = cv2.VideoCapture(video)
  success, image = vidcap.read()
  count = 0
  while success:
    cv2.imwrite("frames/frame%d.jpg" % count, image)
    success, image = vidcap.read()
    count += 1
  vidcap.release()
  return count


def detect_people(dir_path, cas_path, video, frames):
  cap = cv2.VideoCapture(video)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  out = cv2.VideoWriter("tracked.mp4", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width, frame_height))
  cap.release()
  for frame in range(frames):
    img_path="frame"+str(frame)+".jpg"
    body_cascade = cv2.CascadeClassifier(cas_path)
    image = cv2.imread(os.path.join(dir_path, img_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    people, _, _ = body_cascade.detectMultiScale3(
      gray,
      scaleFactor=1.1,
      minNeighbors=1,
      minSize=(50, 110),
      flags=cv2.CASCADE_SCALE_IMAGE,
      outputRejectLevels=True
    )

    for (x, y, w, h) in people:
      cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    out.write(image)
  shutil.rmtree("frames")
  out.release()



#if __name__ == "main":
video_path = sys.argv[1]
casc_path = sys.argv[2]
num_frames = to_frames(video_path)
detect_people("frames", casc_path, video_path, num_frames-1)
