import tensorflow as tf
import logging

def train_valid_generator(data_dir, IMAGE_SIZE, BATCH_SIZE, do_data_augmentation):

    datagenerator_kwagrs = dict(
        rescale = 1./255,
        validation_split = 0.20
    )

    dataflow_kwargs = dict(
        target_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        interpolation = "bilinear"
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
       **datagenerator_kwagrs)

    valid_generator = valid_datagenerator.flow_from_directory(
        directory = data_dir,
        subset = "validation",
        shuffle = False,
        **dataflow_kwargs 
    )

    if do_data_augmentation:
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range = 40,
            horizontal_flip = True,
            width_shift_range = 0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, **datagenerator_kwagrs
        )
        logging.info(f"data augmentation is used for training")
    else:
        train_datagenerator = valid_datagenerator
        logging.inof(f"data augmentation is not used for training")

    train_generator = train_datagenerator.flow_from_directory(
        directory = data_dir,
        subset = "training",
        shuffle = True,
        **dataflow_kwargs 
    )

    return train_generator, valid_generator

