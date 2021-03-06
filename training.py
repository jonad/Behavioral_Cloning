import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout
import pickle
from keras.callbacks import TensorBoard, ModelCheckpoint

BASE_DIR = '/home/workspace/CarND-Behavioral-Cloning-P3/img'

def get_data(filename):
    samples = []
    with open(os.path.join(BASE_DIR, filename)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
     
    samples = samples[1:]
    print('The dataset is {} records'.format(len(samples)))
    
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = BASE_DIR + '/IMG/'+ batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                augmented_image = cv2.flip(image, 1)
                augmented_angle = angle*-1.0
                images.append(image)
                angles.append(angle)
                images.append(augmented_image)
                angles.append(augmented_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)
            
def train(train_samples, validation_samples, callbacks_list):
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3) ))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(24, (5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(36,(5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(48,(5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(64,(3, 3), activation='relu'))
    model.add(Conv2D(64,(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, 
                       steps_per_epoch=len(train_samples),
                       validation_data=validation_generator,
                       validation_steps=len(validation_samples),
                       epochs=5, verbose=1, callbacks=callbacks_list)
    
    with open('./history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
   

    
if __name__ == '__main__':
    filename = 'driving.csv'
    train_samples, validation_samples = get_data(filename)
    keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
    callback_list = [ModelCheckpoint(filepath='model_final.h5',
                                   monitor='val_loss',
                                   save_best_only=True),
                    TensorBoard(log_dir='./logs', histogram_freq=0,  
          write_graph=True, write_images=False)]
    
    train(train_samples, validation_samples, callback_list)
    
    