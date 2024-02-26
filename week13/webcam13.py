import sys

import cv2
import numpy as np

sys.path.append("../..")
from utilities.webcam_control import Webcam

# Parameter
phi = 0  # Hue offset
sp = 0  # Satuartion offset
ip = 0  # Intensity offset
settings = {
    "frame_width": 2048,
    "frame_height": 1536,
    "exposure": -4,
    "gain": 0,
}  # manual camera settings
# settings_file = 'settings.txt'  # file name with settings
# (generated by 'get_settings.py')

downsampling_factor = 1  # downsampling factor (increases speed, but data is lost)

# Open video stream
camera = Webcam(port=0, settings=settings, downscale=downsampling_factor)

while 1:
    ret, frame = camera.get_frame()
    if not ret:
        break

    # Preprocessing
    f = np.array(frame / 255, dtype=np.float32)
    # f = np.rot90(f, k=2)  # rotate image by 180 degrees

    # BGR to HSI conversion and apply offsets to layers
    f_hsi = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    [H, S, I] = np.squeeze(np.split(f_hsi, 3, 2))
    H = np.mod(H + phi, 360)
    phi += 2
    S = np.mod(S + sp, 1)
       # sp = np.mod(sp + 1 + 0.01, 2) - 1
    I = np.mod(I + ip, 1)
       # ip = np.mod(ip + 1 + 0.01, 2) - 1
    f_trans = np.stack([H, S, I], axis=2)

    g = cv2.cvtColor(f_trans, cv2.COLOR_HSV2BGR)

    # Display the resulting frame
    cv2.imshow("original", f)
    cv2.imshow("transformed", g)

# When everything done, release the capture
camera.close()