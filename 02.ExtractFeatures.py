from vit_keras import vit, utils
import numpy
import tensorflow as tf
import cv2
import pickle
import numpy as np
import scipy.io as sio
#from keras.preprocessing import image
import os


image_size = 224
classes = utils.get_imagenet_classes()
model = vit.vit_b16(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)

Dataset= r'F:\VTM\dataset\train/'
savefeatures=r'F:\VTM\code\Features _16frames\trainFeatures/'
#Define dataset path
dataset_directory = Dataset
dataset_folder = os.listdir(dataset_directory)

#Feature extractions
DatabaseFeautres = []
DatabaseLabel = []
nu_class=14

cc=0
mycounter=1
for dir_counter in range(0,len(dataset_folder)):
    cc+=1
    print('Processing class:   ', cc, 'of 14')
    single_class_dir = dataset_directory + "/" + dataset_folder[dir_counter]
    all_videos_one_class = os.listdir(single_class_dir) 
    for single_video_name in all_videos_one_class:        
        video_path = single_class_dir + "/" + single_video_name
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_features = []
        frames_counter = -1
        print('video name =>', single_video_name)
        while(frames_counter < total_frames-1):

            frames_counter+=1
            ret, frame = capture.read()
            if (ret):
                frame = cv2.resize(frame, (224,224))
                img_data = vit.preprocess_inputs(frame).reshape(1, image_size, image_size, 3)
                single_featurevector = model.predict(img_data)
                
                #print('what is the sahpe of single_featurevector =>', single_featurevector.shape)#(1, 7, 7, 1280)
                video_features.append(single_featurevector)
                if frames_counter%15 == 14:
                    temp = np.asarray(video_features)
                    DatabaseFeautres.append(temp)
                    DatabaseLabel.append(dataset_folder[dir_counter])
                    video_features = []
                    #print('[INFO]....complete 30FramesSequnce ===>',mycounter)
                    
                    mycounter+=1

TotalFeatures= []
OneHotArray = []
for sample in DatabaseFeautres:
    #TotalFeatures.append(sample.reshape([1,30000]))
    TotalFeatures.append(sample.reshape([1,15000]))


TotalFeatures = np.asarray(TotalFeatures)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres),15000])

OneHotArray = []
kk=1;
for i in range(len(DatabaseFeautres)-1):
    OneHotArray.append(kk)
    if (DatabaseLabel[i] != DatabaseLabel[i+1]):
        kk=kk+1;

with open("OneHotArray.pickle", 'wb') as f:
  pickle.dump(OneHotArray, f)
    
OneHot=  np.zeros([len(DatabaseFeautres),nu_class], dtype='int');


for i in range(len(DatabaseFeautres)-1):
    print(i)
    OneHot[i,OneHotArray[i]-1] = 1



np.save(savefeatures+'/TrainVIT-16_15Frames_244',TotalFeatures)
sio.savemat(savefeatures+'/Train_LabelsVIT-16_15Frames_244', mdict={'Total_labeL': OneHot})






