# Read a video containing time-resolved data from a 7-segment display and transcribe to a CSV file.
#
#   THE WILSON LAB
#   ajwilsonlab.org
#
#   Created by: Johann Hemmer
#   johann.hemmer(at)louisville.edu
#   4 Dec 2025
#
# Parameters:
#   FILE_PATH:          relative or absolute path to video
#   OUTPUT_CSV:         relative or absolute path to ouput CSV file
#   MINIMUM_VALUE:      recognized values lower than this will be ignored
#   MAXIMUM_VALUE:      recognized values greater than this will be ignored
#   N:                  number of frames to average (see instructions)
#   MAX_DIFF:           maximum absolute difference between consecutive readings, useful for ignoring
#                           read errors if the display doesn't instanly change. Set to a large number
#                           to disable this behavior (e.g., 1e99).
#   REPEAT_IF_DIFF:     see `MAX_DIFF`. if the difference between 2 consecutive is greater than 
#                           MAX_DIFF, setting this to `True` will repeat the last value. Setting it 
#                           to `False` will ignore the the value.
#   FRAME_STEP:         increase to skip every nth frame
#   DESKEW_ANGLE_DEG:   angle in degrees to deskew image
#   CUSTOM_CONFIG:      config for Tesseract
#   PREVIEW:            if `True`, will display a window showing raw and preprocessed ROIs
#
# Instructions:
#   Specify path to the input video and output CSV file and adjust the minimum and maximum values 
#   expected using the respective parameters. This will be the first line of defense against OCR 
#   mistakes. Adjust the number of frames to average by changing the value of `N`. This can help
#   by smoothing the transitions between numbers in the LCD display, which sometimes isn't 
#   instant and causes problems with the detection algorithm. `N` should be low enough so that 
#   the sampling rate is lower than the expected rate of change of the data. While the script is
#   running and if PREVIEW is set to True, you may hit ESC to stop early and save the data.
#
#   After running the script, a window will pop-up showing the first frame of the video. Click
#   and drag a rectangle around the area containing the digits. The area should be large enough
#   to encompass all digits and an extra space around them. Avoid including extra features with
#   varying degrees of contrast as they can be read as numbers unintentionally.
#
# Notes:
#   This implementation of OCR ignores decimal separators, and since the measurements we were
#   doing at the time were of degree Celsius in the ~20s range, I hard coded a multiplication
#   by 10 

# ==== CONFIG ====
FILE_PATH = r"C:\Users\jhemmer\OneDrive - University of Louisville\0. Lab\4. Projects\EM-field mass transfer\R1 Data\Image Recognition\blue_1min.MOV"
OUTPUT_CSV = "output/test.csv"
MINIMUM_VALUE = 20
MAXIMUM_VALUE = 26
N = 15
MAX_DIFF = 0.11
REPEAT_IF_DIFF = False
FRAME_STEP = 1
DESKEW_ANGLE_DEG = 3
CUSTOM_CONFIG = r'-l ssd --psm 13 -c tessedit_char_whitelist="0123456789"'
PREVIEW = True
# ================

## Imports
import cv2
import pandas as pd
import pytesseract
import numpy as np

## Functions
def preprocess(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Adjust contrast
    alpha = 1.0
    beta = -20
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Smoothing
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def deskew(img, angle_deg):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def read_value(roi_bin, previous: None | float = None):
    text = pytesseract.image_to_string(roi_bin, config=CUSTOM_CONFIG)
    text = text.strip()

    try:
        val = float(text)/10 # see notes in preamble

        # Ignore values greater or smaller than the minimum
        if (val < MINIMUM_VALUE) or (val > MAXIMUM_VALUE):
            print(f"Value ignored, outside bounds: {val}")
            return None
        else:
            if previous:
                if abs(val - previous) > 0.11:
                    print(f"Sudden change detected: {val} from {previous}.")
                    if REPEAT_IF_DIFF:
                        return previous
                    else:
                        return None
                else:
                    return val
            else:
                return val

    except ValueError:
        return None

def average_frames(frame_list):
    if not frame_list:
        return None
    stack = np.stack([f.astype(np.float32) for f in frame_list], axis=0)
    avg = np.mean(stack, axis=0)
    return avg.astype(np.uint8)

## Main
# Open and read video
video = cv2.VideoCapture(FILE_PATH)

fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

ret, first_frame = video.read()

# Open the first frame to select the ROI
sel_roi_name = "Select thermometer ROI"
roi = cv2.selectROI(sel_roi_name, first_frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow(sel_roi_name)

x, y, w, h = roi

print(f"Using ROI: x = {x}, y = {y}, w = {w}, h = {h}")

# Reset video to the beginning
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Main loop
data = []
frame_idx = 0
buffer = []
prev_value = None
nones = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    if frame_idx % FRAME_STEP != 0:
        frame_idx += 1
        continue

    roi_frame = frame[y:y+h, x:x+w]
    roi_frame = deskew(roi_frame, DESKEW_ANGLE_DEG)

    buffer.append(roi_frame)

    if len(buffer) == N:
        avg_roi = average_frames(buffer)
        buffer.clear()

        roi_bin = preprocess(avg_roi)
        value = read_value(roi_bin, previous=prev_value)
        prev_value = value

        if value is not None:
            t_sec = frame_idx / fps
            data.append({"time_s": t_sec, "val": value})
            print(f"Read value: {value} at t = {t_sec:.2f} s")
        else:
            nones += 1

        # Display preview
        if PREVIEW:
            raw_bgr = cv2.cvtColor(avg_roi, cv2.COLOR_GRAY2BGR) if len(avg_roi.shape)==2 else avg_roi # type: ignore
            bin_bgr = cv2.cvtColor(roi_bin, cv2.COLOR_GRAY2BGR)
            combined = cv2.hconcat([raw_bgr, bin_bgr]) # type: ignore
            cv2.imshow("ROI (raw -> preprocessed)", combined)

    # Hit ESC to stop early
    key = cv2.waitKey(1)
    if key == 27:
        break

    frame_idx += 1

video.release()

# Save
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_CSV}.")
print(f"Number of failed or discarded readings: {nones}")
