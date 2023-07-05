
import mediapipe as mp
import time
import numpy as np
import cv2
import onnx
import onnxruntime
import tensorflow as tf
import torch
import pyvirtualcam
import argparse
import math
from spellchecker import SpellChecker

spell = SpellChecker()

parser = argparse.ArgumentParser(description="Sign Language Detection and Captioning")
parser.add_argument("--model", default="mobilenetv2",
  help="Model to be Used")
args = parser.parse_args()

dims = 0

## ARGPARSER CHECK MODEL NAMES AND LOCATION ##

if args.model == "mobilenetv2":
    dims = (1, 224, 224, 3)
    model="saved_model-mb2_bs-32_dr-0.2_lr-0.0001_e50_fds-2.onnx"

elif args.model == "mnist":
    dims = (1, 28, 28, 1)
    model="saved_model-mnist.onnx"
elif args.model == "alexnet":
    dims = (1, 227, 227, 3)
    model="saved_model-alexnet.onnx"

num_letter = 15
max_letter_label = 15 
threshold = 0.7
clear_time = 3
text_color = (0, 255, 0)
font = cv2.FONT_HERSHEY_TRIPLEX

def mnist_process(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mnist_img = cv2.resize(img, (28,28))
	mnist_img = mnist_img/255
	mnist_img = mnist_img.reshape(-1,28,28,1).astype(np.float32)
	return mnist_img

def alexnet_process(img):
    alexnet_img = cv2.resize(img, (227,227))
    alexnet_img = alexnet_img/255
    alexnet_img = alexnet_img.reshape(-1,227,227,3).astype(np.float32)
    return alexnet_img

device = torch.device('cpu')

# EDIT DEPENDING ON MODEL RUN
data = torch.randn(dims)
data = data.to(device=device)
data = data.detach().cpu().numpy() if torch.Tensor.requires_grad else torch.Tensor.cpu().numpy()

so = onnxruntime.SessionOptions()
so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL


onnx.checker.check_model(model)

session = onnxruntime.InferenceSession(model, so)
options = session.get_provider_options()

input_name = session.get_inputs()[0].name


#Warm Up
session.run(None, {input_name: data} )


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

offset = 50
imgSize = 240

labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 
7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
14: 'O',15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

#function to find the letter with the highest number of occurrence on frame slice
def most_frequent_letter(letter_label):
    return max(set(letter_label), key = letter_label.count)

#function to find the number of occurrence of most frequent letter on frame slice
def num_of_occurrence(letter_label, max_letter):
    return letter_label.count(max_letter)

cap = cv2.VideoCapture(0)

#set the initial time for previous frame when the program runs
old_time = 0

padding = 50
start = time.time()
counter = 0
display_time = 2
fps = 0
frames = 0
start_new_word = False
auto_corrected = 0

with mp_hands.Hands(model_complexity=0,
  					min_detection_confidence = 0.2, 
                    min_tracking_confidence = 0.3,
                    max_num_hands = 1,
                    static_image_mode= False) as hands:
    
    fmt = pyvirtualcam.PixelFormat.BGR
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print(height, width)

    with pyvirtualcam.Camera(width=int(width), height=int(height), fps=30, fmt=fmt) as cam:
        
        while True:
            #mp_hands = mp.solutions.hands
            hands = mp_hands.Hands()
            #mp_drawing = mp.solutions.drawing_utils
            hand_present = False

            letter_label = []
            letter_bank = ""
            
            while cap.isOpened():
                success, image = cap.read()
                image = cv2.flip(image, 1)
                framergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
                try:
                    imgCopy = image.copy()
                    h, w, c = image.shape
                    fr = 1 # displays the frame rate e

                    image.flags.writeable = False
                    results = hands.process(framergb)
                    image.flags.writeable = True
                    hand_landmarks = results.multi_hand_landmarks
        
                    #### 		  Draw Square Bounding Box Region		  ####        
        
                    if hand_landmarks:
                        for handlmks in hand_landmarks:
                            x_max = 0
                            y_max = 0
                            x_min = w
                            y_min = h
                            
                            for lm in handlmks.landmark:
                                x, y = int(lm.x * w), int(lm.y * h)
                                
                                if x > x_max:
                                    x_max = x
                                if x < x_min:
                                    x_min = x
                                if y > y_max:
                                    y_max = y
                                if y < y_min:
                                    y_min = y
                                
                            x_diff = (x_max - x_min)
                            y_diff = (y_max - y_min)
                            
                            if x_diff > y_diff:
                                pad = (x_diff - y_diff)//2
                                y_min -= pad 
                                y_max += pad
                                curr_img = imgCopy[y_min - padding: y_max + padding, x_min - padding:x_max + padding]
        
                            if y_diff > x_diff:
                                pad = (y_diff - x_diff)//2
                                x_min -= pad 
                                x_max += pad 
                                curr_img = imgCopy[y_min - padding: y_max + padding, x_min - padding:x_max + padding]
                            
                            # resized_img = cv2.resize(curr_img, (28,28), interpolation = cv2.INTER_AREA)
    
            
                            cv2.rectangle(image, (x_min - padding, y_min - padding), 
                                        (x_max + padding, y_max + padding), (0, 255, 0), 2)
        
        
                            ## Processing Cropped Hand Region
                            # data = np.empty((1, 224, 224, 1)).astype(np.float32)
                            # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        
                            # data[0] = resized_img
        
                            ## Preprocess
                            if args.model == "mobilenetv2":
                                resized_img = cv2.resize(curr_img, (224,224), interpolation = cv2.INTER_AREA)
                                data = np.empty((1, 224, 224, 3)).astype(np.float32)
                                data[0] = resized_img
                                proc_img = tf.keras.applications.mobilenet_v2.preprocess_input(data)
                            elif args.model == "mnist":
                                proc_img = mnist_process(curr_img)
                            elif args.model == "alexnet":
                                proc_img = alexnet_process(curr_img)

        
                            ## Passing the processed image/frame to the model and making inference
                            result = session.run(None, {input_name: proc_img})
                            prediction=(np.argmax(np.array(result).squeeze(), axis=0))

                            letter = labels[prediction]

                            ## Drawing the Predicted Text to the OpenCV Window
                            cv2.putText(image, letter, (x_min - padding, y_min - padding - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)

                            #adds the letter to letter bank
                            letter_label.append(letter)
                            
                            if len(letter_label) >= max_letter_label:
                                #print('im in')
                                #retrieves the letter with the highest frequency
                                max_letter = most_frequent_letter(letter_label)
                                
                                if((num_of_occurrence(letter_label, max_letter)/max_letter_label) >= threshold):
                                    #clears letter bank if it exceeds or reaches the maximum number of letters
                                    print(letter_bank)
                                    if(len(letter_bank) >= num_letter):
                                        #print('=====letter bank cleared====')
                                        letter_bank = ""
                                    #adds the most frequent letter in letter bank
                                    letter_bank += max_letter
                                    print(letter_bank)
                                letter_label.clear()
                            #changes boolean to True since hand is present
                            hand_present = True

                    else:
                        #clears frame slice if hand is absent on the screen
                        letter_label.clear()
                        if(hand_present == True):
                            #gets the time of last frame processed
                            old_time = time.time()
                        #changes boolean to False when hand is absent
                        #print('No hand detected')
                        hand_present = False


                    ## Autocorrect ##
                    if(hand_present == False):
                        cv2.putText(image, f'No Hand Detected', (180, 420),
                                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
                    else: 
                        misspelled = list(spell.unknown([letter_bank]))

                        corrected = " "
                        
                        if len(misspelled) != 0:
                            corrected = spell.correction(misspelled[0])
                            auto_corrected = 1
                        else:
                            auto_corrected = 2
                        
                        if auto_corrected == 1:
                            cv2.putText(
                                    image,
                                    corrected,
                                    (180, 420),
                                    font, 2.0, (0, 255, 255), 2) 
                        elif auto_corrected == 2: # if correct!
                            cv2.putText(
                                    image,
                                    corrected,
                                    (180, 420),
                                    font, 2.0, text_color, 2)
                        else:
                            cv2.putText( 
                                    image,
                                    corrected,
                                    (180, 420),
                                    font, 2.0, (255, 255, 255), 2)

                    #displays the letter bank
                    cv2.putText(image, letter_bank, (180, 360), cv2.FONT_HERSHEY_TRIPLEX, 2.0, text_color, 2)
                    
                    t = time.time()
                    #clears letter bank after clear time when hand is absent
                    if((t - old_time) > clear_time) and (not hand_present):
                        letter_bank = ""

                    counter+=1
                    frames +=1
                    
                    frametime = time.time()
                    totalTime = frametime-start
                    if totalTime >=2:
                        fps = counter/totalTime
                        counter = 0
                        start = frametime

                    if frames == 300:
                        print(frames, fps)
        
                    #cv2.putText(image, f'FPS: {"%.2f" % fps}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)
                    # cv2.putText(image, f'aveFPS: {int(fps_ave)}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cam.send(cv2.flip(image,1))
                    cam.sleep_until_next_frame()
                    cv2.imshow('MediaPipe Hands', image)
        
                except Exception as e:
                    print(str(e))

                key = cv2.waitKey(1)
                
                #Close all Windows
                if key == ord("1"):
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break