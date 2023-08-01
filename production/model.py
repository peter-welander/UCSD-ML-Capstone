import os.path
import pathlib

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


_BATCH_SIZE = 32
_IMG_SIZE = 192
_BASE_MODEL_URL = 'https://tfhub.dev/google/bit/s-r50x1/1'
_CHECKPOINT_PATH = 'checkpoints/cp.ckpt'

def load_base_model():
    return hub.KerasLayer(_BASE_MODEL_URL)

def Model(num_classes, load_checkpoint=False):
    base_model = load_base_model()
    
    model = tf.keras.Sequential([
        layers.Rescaling(1./255),
        base_model,
        layers.Dropout(0.3),
        layers.Dense(num_classes),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    if load_checkpoint:
        model.load_weights(_CHECKPOINT_PATH)

    model.build((None, _IMG_SIZE, _IMG_SIZE, 3))
    return model


def Fit(model, train_ds, val_ds,
        epochs=10,
        checkpoint_path = _CHECKPOINT_PATH):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback],
    )
    
def OpenDataset(images_dir):
    data_dir =  pathlib.Path(os.path.abspath(images_dir))
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(_IMG_SIZE, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
    )
    return train_ds, val_ds
