"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
from collections import deque
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters)

    coordinates = detection(test_img)
    
    names = recognition(test_img,coordinates)

    results = []
    for i in range(len(coordinates)):
        results.append({"bbox": coordinates[i], "name": names[i]})
    
    return results
    #raise NotImplementedError
    

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    #Using SIFT operation to extract and list features

    descriptor_list = {}
    sift = cv2.SIFT_create()

    for element in characters:
        if element[0]=='dot':
            continue
        keypoints,descriptors = sift.detectAndCompute(element[1],None)
        descriptor_list[element[0]] = descriptors.tolist()

    with open('desc.json', 'w') as descwr:
        json.dump(descriptor_list, descwr)


    #raise NotImplementedError

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    
    th = np.zeros(test_img.shape)
    rows, cols = test_img.shape
    for i in range(rows):
        for j in range(cols):
            if test_img[i][j] < 132:
                th[i][j] = 1
            else: 
                th[i][j] = 0

    coord_detected = []
    nodes_visited = set()

    #Implementing DFS to traverse through 
    def dfs(x,y):
        dfsQ = deque()
        dfsQ.append((x,y))
        X_coord,Y_coord,Width,Height = 5000,5000,-1000,-1000
        while dfsQ:
            i,j = dfsQ.pop()
            X_coord = min(X_coord,j)
            Y_coord = min(Y_coord,i)
            Width = max(j-X_coord,Width)
            Height = max(i-Y_coord,Height)
            nodes_visited.add((i,j)) 
            for a,b in [(-1,0), (1,0), (0,1), (0,-1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                if 0<=i+a<rows and 0<=j+b<cols and (i+a,j+b) not in nodes_visited and th[i+a][j+b]==1:
                    dfsQ.append((i+a,j+b))
        return [X_coord,Y_coord,Width+1,Height+1]

    for i in range(rows):
        for j in range(cols):
            if (i,j) not in nodes_visited and th[i][j] == 1:
                coord_detected.append(dfs(i,j))
            else:
                continue
    
    return coord_detected
    #raise NotImplementedError

#Creating a feature matching function which uses Sum of Squared Differences(SSD) division.
#desc1 desc2 are features from the test image and given characters respectively
def feature_matching(desc1,desc2):
    dist = np.zeros((len(desc1),len(desc2)))
    for i in range(desc1.shape[0]):
        for j in range(desc2.shape[0]):
            ssd = np.sum(np.square(desc1[i] - desc2[j]))
            dist[i][j] = ssd 

    #array to store the matches
    best_matches = []

    for i in range(desc1.shape[0]):
        idx = np.argsort(dist[i])
        #f2 stands for best match and sec_f2 stands for second best match
        f2,sec_f2 = idx[0],idx[1]
        if dist[i][f2] / dist[i][sec_f2] < 0.35 :
            best_matches.append(f2)

    return best_matches
  
    

def recognition(test_img,boundary):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    #reading features detected from the enrollment function
    with open('desc.json', "r") as descrd:
        descriptors = json.load(descrd)

    pred_names = [] #.

    #Using SIFT to extract features(desc1) from the test image and matching them with the features(desc2) extracted in enrollment
    sift = cv2.SIFT_create()

    for bbox in boundary:
        X,Y,W,H = bbox
        img = test_img[Y:Y+H,X:X+W].astype('uint8')
        key1,desc1 = sift.detectAndCompute(img,None)

        if len(key1) == 0:
            pred_names.append('dot')
            continue

        #array to store all matches
        matches = []
        for elements in descriptors:
            desc2 = np.array(descriptors[elements])
            matches.append((elements,feature_matching(desc1,desc2)))
        matches.sort(key = lambda X: len(X[1]), reverse = True)

        if len(matches[0][1]) > 0:
            pred_names.append(matches[0][0])
        else:
            pred_names.append("UNKNOWN")
    return pred_names

    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
