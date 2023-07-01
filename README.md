# Sign Language Detection and Captioning for Video Conferencing Using Machine Learning
An Undergraduate Student Project aiming to develop an add-on for videoconferencing platforms to detect and caption sign language gestures.


|File/Folder | Description |
| --- | --- |
| Building Codes | Contains the Google Colab Notebooks used to build and train the models |
| Models | Contains the trained models for MobileNetV2, AlexNet, and MNIST (Due to the large size, the AlexNet Model as well as all the other models are available [here](https://drive.google.com/drive/folders/1OVVqtgaf-XeDgLGElr0gqtXv2KwAHjHj?usp=sharing) |
| main.py | is the main program for this project |
| requirements.txt | Project required dependencies |
  
## Prerequisite/Setup
1. Download and Install Python and Jupyter Notebook 
2. Download and install **[OBS Studio](obsproject.com)** and its **[Virtual Camera Plug-in](https://obsproject.com/forum/resources/obs-virtualcam.949/)** in the same directory.
3. Setup **OBS Studio**. You can follow this **[Youtube tutorial](https://youtu.be/fkKC1uSFeCo)**. 
4. Clone the Repository
5. Download necessary libraries/modules/packages using `pip install -r requirements.txt`
6. Modify the model directories by changing the _model_ variable 
7. Run the main program by following the instructions below:
  
## Code Execution
To run the .py files, enter any of the following to the device's console :\
    `python main.py --model mobilenetv2` when using the mobilenetv2 model\
    `python main.py --model mnist` when using the MNIST model\
    `python main.py --model alexnet` when using the AlexNet model

You can also simply run :
    `python main.py` as the program defaults to the mobilenetv2 model
    
## Link/Reference
This project is based on the following works:
* [MNIST Model](https://github.com/chenson2018/APM-Project/blob/master/Final%20Materials/Static_Signs.ipynb?fbclid=IwAR1l7eApNeIa1lXFTH69hKjKG_qFd_WIacZY3FXmvuffWzT3zvx0IUcBEf8)
* [Alexnet Model](https://github.com/vagdevik/American-Sign-Language-Recognition-System/tree/master/2_AlexNet)
* [Bird Classification using EfficientNetB0 Model](https://www.kaggle.com/code/vencerlanz09/bird-classification-using-cnn-efficientnetb0/notebook?scriptVersionId=120482933)

This project also serves as a follow-up improvement to a previous project on American Sign Language Captioning
* [Integrated Visual-Based ASL Captioning in Videoconferencing Using CNN](https://ieeexplore.ieee.org/abstract/document/9977526)
* [Integrated Visual-Based ASL Captioning in Videoconferencing Using CNN (GITHUB)](https://github.com/J-Rikk/asl-captioning/tree/main)
