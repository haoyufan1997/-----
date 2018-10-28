# coding=utf-8
import pandas as pd
import numpy as np
import os
import random
import cv2
import datetime
from tqdm import tqdm
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.layers import normalization 
from keras.applications.densenet import DenseNet169, DenseNet121
from keras.applications.densenet import preprocess_input

from keras import backend as K
K.set_learning_phase(1)

EPOCHS = 40
RANDOM_STATE = 0
learning_rate = 1e-3
TRAIN_DIR = r'../data/processed_train'
VALID_DIR = r'../data/processed_valid'
def get_callbacks(filepath, patience=2):
    def scheduler(epoch):   
        if epoch % 10 == 0 and epoch != 0:       
            lr = K.get_value(SGD.lr)       
            K.set_value(SGD.lr, lr * 0.2)     
            print("lr changed to {}".format(lr * 0.2))    
        return K.get_value(SGD.lr)
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=5e-6, patience=patience, verbose=1, min_lr = 1e-6)
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience*3+4, verbose=1, mode='auto')
    return [lr_reduce, msave, earlystop]

def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x) 
    predictions = Dense(nb_classes, kernel_regularizer=regularizers.l2(0.01),activation='softmax')(x) 
    model = Model(input=base_model.input, output=predictions)
    return model
def get_model(INT_HEIGHT, IN_WIDTH):
    '''
    获得模型
    '''

    base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(INT_HEIGHT,IN_WIDTH, 3))
    model = add_new_last_layer(base_model, 12)
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model

def train_model(save_model_path, BATCH_SIZE, IN_SIZE):

    INT_HEIGHT = IN_SIZE[0]
    IN_WIDTH = IN_SIZE[1]

    callbacks = get_callbacks(filepath=save_model_path, patience=3)
    model = get_model(INT_HEIGHT, IN_WIDTH)

    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input, 
        rotation_range = 60,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        zoom_range = 0.2
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        directory = TRAIN_DIR, 
        target_size = (INT_HEIGHT, IN_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode = 'categorical',
        seed=RANDOM_STATE,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory = VALID_DIR,
        target_size = (INT_HEIGHT, IN_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode = 'categorical',
        seed=RANDOM_STATE,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    model.fit_generator(
        train_generator, 
        steps_per_epoch = 2*(train_generator.samples // BATCH_SIZE + 1), 
        epochs = EPOCHS,
        max_queue_size = 100,
        workers = 1,
        verbose = 1,
        validation_data = valid_generator, 
        validation_steps = valid_generator.samples // BATCH_SIZE,  
        callbacks = callbacks

        )
#
filename = []
filelabel = []
classname = []
def predict(weights_path, IN_SIZE):
    '''
    对测试数
    
    据进行预测
    '''

    INT_HEIGHT = IN_SIZE[0]
    IN_WIDTH = IN_SIZE[1]
    K.set_learning_phase(0)

    test_pic_root_path = r'../data/test'

    for i in sorted(os.listdir(TRAIN_DIR)):
        classname.append(i)
    model = get_model(INT_HEIGHT, IN_WIDTH)
    model.load_weights(weights_path)
    for image in os.listdir(test_pic_root_path):
        pic_path = os.path.join(test_pic_root_path,image)
        img = load_img(pic_path, target_size=(INT_HEIGHT, IN_WIDTH, 3),interpolation='antialias')
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0].tolist()#输出概率的数组，顺序按照keras读取的排序
        print(prediction)
        index = prediction.index(max(prediction))#获得概率最大的类别的索引
        label = classname[index]
        print(label)
        filename.append(image)
        filelabel.append(label)
    submission = pd.DataFrame({'filename': filename, 'label': filelabel})
    submission.to_csv('submit.csv' , header=None, index=False)
    return
        


batch_size = 8
in_size = [550,600] 
weights_path = 'crossloss.h5'

#train_model(save_model_path=weights_path, BATCH_SIZE=batch_size, IN_SIZE=in_size)  
predict(weights_path=weights_path, IN_SIZE=in_size)
