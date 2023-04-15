"""
    This file contains a tensorflow implementation of VGG as reffered to 
    in the paper: https://arxiv.org/pdf/1409.1556.pdf
    This implementation is tested on a tor dataset: Intel Image Classification Dataset ( https://www.kaggle.com/datasets/puneet6060/intel-image-classification )
"""

import logging
import os

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


class VGG:
    """
        Class with all functions realted to VGG model
    """
    def __init__(self, batch_size=16) -> None:
        logging.info("Initialising VGG model")
        self.batch_size = batch_size
        self.class_names = []
    
    def decode_img(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.image.resize(img, (224, 224))
    
    def process_path(self, file_path):
        logging.debug(f"File path being processed: {file_path }")
        parts = tf.strings.split(file_path, os.path.sep)
        logging.debug(f"Parts: {parts}")

        label = parts[-2] == self.class_names
        logging.debug(f"Label: {label}")
        img = self.decode_img(file_path)

        return img, label


    def load_data(self, dataset_path):
        logging.info( f"Starting to load data from {dataset_path}" )
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        if not len(self.class_names):
            self.class_names = [cls for cls in os.listdir(dataset_path) if cls not in self.class_names]
        
        files = tf.data.Dataset.list_files( [os.path.join(dataset_path, "*", "*.jpg"), os.path.join(dataset_path, "*", "*.png")],shuffle=True )
        logging.debug(f"Number of files read: {len(files)}")

        dataset = files.map(self.process_path, num_parallel_calls=AUTOTUNE).batch(self.batch_size).prefetch( buffer_size=AUTOTUNE )

        return dataset


    def build_model(self, num_classes, model_name="vgg-16"):
        logging.info( f"Building {model_name}" )
        model = Sequential( name="VGG-16" )

        model.add( layers.Input( (224, 224, 3) ) )
        
        # Block 1
        model.add( layers.Conv2D( 64, 3, activation="relu", padding="same", name="Conv1_1" ) )
        model.add( layers.Conv2D( 64, 3, activation="relu", padding="same", name="Conv1_2" ) )
        model.add( layers.MaxPool2D((2, 2), name="MaxPool_1") ) 

        # Block 2
        model.add( layers.Conv2D( 64, 3, activation="relu", padding="same", name="Conv2_1" ) )
        model.add( layers.Conv2D( 128, 3, activation="relu", padding="same", name="Conv2_2" ) )
        model.add( layers.MaxPool2D((2, 2), name="MaxPool_2") ) 

        # Block 3
        model.add( layers.Conv2D( 256, 3, activation="relu", padding="same", name="Conv3_1" ) )
        model.add( layers.Conv2D( 256, 3, activation="relu", padding="same", name="Conv3_2" ) )
        model.add( layers.MaxPool2D((2, 2), name="MaxPool_3") ) 

        # Block 4
        model.add( layers.Conv2D( 512, 3, activation="relu", padding="same", name="Conv4_1" ) )
        model.add( layers.Conv2D( 512, 3, activation="relu", padding="same", name="Conv4_2" ) )
        model.add( layers.MaxPool2D((2, 2), name="MaxPool_4") ) 

        # Block 5
        model.add( layers.Conv2D( 512, 3, activation="relu", padding="same", name="Conv5_1" ) )
        model.add( layers.Conv2D( 512, 3, activation="relu", padding="same", name="Conv5_2" ) )
        model.add( layers.MaxPool2D((2, 2), name="MaxPool_5") ) 

        # Fully Connected layers
        model.add( layers.Flatten() )
        model.add( layers.Dense(4096, activation="relu", name="FC_1") )
        model.add( layers.Dense(4096, activation="relu", name="FC_2") )
        model.add( layers.Dense(num_classes, activation="relu", name="FC_3") )
        
        return model

    def fit(self, train_path=None, val_path=None, X = None, y = None, eps=5):
        if train_path and val_path:
            train_dataset = self.load_data(train_path)
            val_dataset = self.load_data(val_path)
        elif X and y:
            logging.error( f"Model training with input as numpy arrays is not supported yet!!! Sorry for the inconvience😞😞 " )
            raise Exception()
        else:
            logging.error( f"Please provid a valid input" )
            raise Exception()
        
        model = self.build_model(num_classes = len(self.class_names) )

        model.compile( optimizer=Adam(learning_rate=1e-04), loss=CategoricalCrossentropy(), metrics=['accuracy'] )
        model.summary()
 
        history = model.fit( train_dataset, validation_data=val_dataset, epochs=eps )

        if not os.path.exits("output"):
            os.makedirs("output")
        model.save(r"output/model.h5")


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s - %(filename)s.%(funcName)s: %(message)s', level=logging.DEBUG )

    ROOT = r"data\Intel Image Classification"
    train_dir = os.path.join(ROOT, r"train")
    val_dir = os.path.join(ROOT, r"val")
    test_dir = os.path.join(ROOT, r"test")

    with tf.device("/device:gpu:0"):
        vgg_model = VGG().fit( train_path=train_dir, val_path=val_dir )