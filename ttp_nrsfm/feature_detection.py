'''
This file contains code for feature detection using ORB descriptors and matching with BFMatcher.
The correspondece matrix is also generated here
Author: Mitterrand Ekole
Date:07-02-2020
'''

import cv2
import numpy as np


def detect_compute(images):
    ''' Takes a list of images and returns keypoints and descriptors for each image in the list'''
    orb=cv2.ORB_create()
    keypoints=[]
    descriptors=[]

    for image in images:
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoint, descriptor=orb.detectAndCompute(gray, None)

        keypoints.append(keypoint)
        descriptors.append(descriptor)

    return keypoints, descriptors


def match_keypoint(descriptors):
    ''' Takes a list of descriptors and returns a list of matches between consecutive images'''
    matches=[]
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #match keypoints between consecutive images
    for i in range(len(descriptors)-1):
        matches.append(bf.match(descriptors[i], descriptors[i+1]))
    
    return matches

def create_corres_matrix(matches,num_images, num_corres):
    ''' Takes in matches, num_images in batch and number of correspondences and returns a correspondence matrix'''

    corres_mat=np.zeros((num_corres, num_images), dtype=int)

    for i, match in enumerate(matches):
        #sort matches based on distance
        match=sorted(match, key=lambda x:x.distance)

        #get the first num_corres matches
        match=match[:num_corres]

        #extract indices of the matches
        indices=[mat.queryIdx for mat in match]

        #populate the correspondence matrix
        corres_mat[:,i]=indices

    return corres_mat



       