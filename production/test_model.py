import tensorflow as tf
import numpy as np
from model import Model, Fit, OpenDataset
from PIL import Image
from unittest.mock import patch
from tensorflow.keras.layers import Flatten

def RandomImage():
    img_array = np.random.rand(192,192,3) * 255
    image = tf.cast(img_array, tf.float32)
    image = tf.image.resize(image, [192, 192])
    image = np.expand_dims(image, axis = 0)
    return image

_CHECKPOINT_PATH = 'checkpoints/unit_test.ckpt'
IMAGES_DIR = "../images/Flowers5/"
IMG_SIZE = 192


class ModelTest(tf.test.TestCase):

    def setUp(self):
        super(ModelTest, self).setUp()
        np.random.seed(42)

    @patch('model.load_base_model')
    def test_create_model(self, mock_data_loader):
        mock_data_loader.return_value = Flatten()
        model = Model(101)
        mock_data_loader.assert_called()

    @patch('model.load_base_model')
    def test_build_model(self, mock_data_loader):
        mock_data_loader.return_value = Flatten()
        model = Model(101)

        model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))

    @patch('model.load_base_model')
    def test_model_output_shape(self, mock_data_loader):
        mock_data_loader.return_value = Flatten()
        image = RandomImage()
        num_classes = 5
        want_shape = (1, num_classes)

        model = Model(num_classes)
        prediction = model.predict(image)

        self.assertShapeEqual(
            prediction,
            np.zeros(shape=want_shape),
        )

    @patch('model.load_base_model')
    def test_fit(self, mock_data_loader):
        mock_data_loader.return_value = tf.keras.layers.Flatten()
        train_ds, val_ds = OpenDataset(IMAGES_DIR)
        num_classes = len(train_ds.class_names)

        model = Model(num_classes)
        Fit(model, train_ds, val_ds, epochs=1, checkpoint_path=_CHECKPOINT_PATH)

        mock_data_loader.assert_called()

if __name__ == '__main__':
    tf.test.main()
