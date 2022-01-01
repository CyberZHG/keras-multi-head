import os
import tempfile
import unittest

import numpy as np
from tensorflow import keras

from keras_self_attention import SeqSelfAttention as Attention
from keras_multi_head import MultiHead


class MaskFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, _, mask=None):
        return mask


class TestMultiHead(unittest.TestCase):

    @staticmethod
    def data_generator(batch_size=32):
        while True:
            max_len = np.random.randint(5, 10)
            data = np.zeros((batch_size, max_len))
            tag = np.zeros(batch_size, dtype='int32')
            for i in range(batch_size):
                datum_len = np.random.randint(1, max_len - 1)
                for j in range(datum_len):
                    data[i, j] = np.random.randint(1, 4)
                tag[i] = int(1 in data[i])
            yield data, tag

    def test_multi_attention(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=5, output_dim=3, mask_zero=True, name='Embed'))
        model.add(MultiHead(Attention(name='Attention'), layer_num=5, name='Multi-Head-Attention'))
        model.add(keras.layers.TimeDistributed(MaskFlatten(), name='Flatten'))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(units=8), name='Bi-GRU'))
        model.add(keras.layers.Dense(units=2, activation='softmax', name='Dense'))
        model.build()
        model.compile(
            optimizer='adam',
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=[keras.metrics.sparse_categorical_accuracy],
        )

        model.fit_generator(
            generator=self.data_generator(),
            steps_per_epoch=100,
            epochs=100,
            validation_data=self.data_generator(),
            validation_steps=10,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
            ],
        )
        model.layers[1].set_weights(model.layers[1].get_weights())

        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'MaskFlatten': MaskFlatten,
            'SeqSelfAttention': Attention,
            'MultiHead': MultiHead,
        })
        model.summary()
        for data, tag in self.data_generator():
            predicts = model.predict(data)
            predicts = np.argmax(predicts, axis=-1)
            self.assertGreaterEqual(np.sum(tag == predicts), 30)
            break

    def test_multi_cnn(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=5, output_dim=3, name='Embed'))
        model.add(MultiHead([
            keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', name='Conv1'),
            keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', name='Conv2'),
            keras.layers.Conv1D(filters=32, kernel_size=7, padding='same', name='Conv3'),
        ], name='Multi-Head-Attention'))
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten(), name='Flatten'))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(units=8), name='Bi-GRU'))
        model.add(keras.layers.Dense(units=2, activation='softmax', name='Dense'))
        model.build()
        model.compile(
            optimizer='adam',
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=[keras.metrics.sparse_categorical_accuracy],
        )

        model.fit_generator(
            generator=self.data_generator(),
            steps_per_epoch=100,
            epochs=100,
            validation_data=self.data_generator(),
            validation_steps=10,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
            ],
        )
        model.layers[1].set_weights(model.layers[1].get_weights())

        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'MultiHead': MultiHead})
        model.summary()
        for data, tag in self.data_generator():
            predicts = model.predict(data)
            predicts = np.argmax(predicts, axis=-1)
            self.assertGreaterEqual(np.sum(tag == predicts), 30)
            break
