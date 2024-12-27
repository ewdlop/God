import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to compute histogram similarity
def histogram_similarity(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to compute SSIM
def compute_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# Function to compare keypoints
def keypoint_matching(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

# Load images
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

# Resize to same dimensions
image1 = cv2.resize(image1, (500, 500))
image2 = cv2.resize(image2, (500, 500))

# Compute similarities
hist_sim = histogram_similarity(image1, image2)
ssim_score = compute_ssim(image1, image2)
keypoint_matches = keypoint_matching(image1, image2)

print(f"Histogram Similarity: {hist_sim:.2f}")
print(f"SSIM Score: {ssim_score:.2f}")
print(f"Keypoint Matches: {keypoint_matches}")
