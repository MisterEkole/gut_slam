import cv2
import numpy as np
from feature_detection import detect_compute, match_keypoint, create_corres_matrix
import matplotlib.pyplot as plt
import glob

def display_keypoints(image, keypoints, title="Keypoints"):
    # Display image with keypoints
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def display_matches(image1, image2, keypoints1, keypoints2, matches, title="Matches"):
    # Display matches between two images
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def display_correspondence_matrix(corres_mat):
    # Display correspondence matrix
    plt.figure()
    plt.imshow(corres_mat, cmap='gray')
    plt.title("Correspondence Matrix")
    plt.xlabel("Image Index")
    plt.ylabel("Correspondence Index")
    plt.colorbar()
    plt.show()

def main():
    image_path='data/*.jpg' #path to dataset
    #read list of images file paths 
    images=glob.glob(images)

    #read images and convert to list
    images=[cv2.imread(image) for image in images]


    #detect and compute keypoints and descriptors
    keypoints, descriptors=detect_compute(images)
    matches=match_keypoint(descriptors)
    corres_mat=create_corres_matrix(matches, len(images), num_corres=10)
    # Display keypoints for the first image
    display_keypoints(images[0], keypoints[0], title="Keypoints - Image 1")

    display_matches(images[0], images[1], keypoints[0], keypoints[1], matches[0], title="Matches - Image 1 vs Image 2")

    # Display the correspondence matrix
    display_correspondence_matrix(corres_mat)

if __name__ == "__main__":
    main()