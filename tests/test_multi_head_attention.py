import os
import tempfile
import unittest
import numpy as np
from keras_self_attention.backend import keras
from keras_self_attention.backend import backend as K
from keras_multi_head import MultiHeadAttention
from .multi_head_attention_brute import MultiHeadAttentionBrute


class GetMask(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(GetMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def call(self, inputs, mask=None, **kwargs):
        return K.cast(mask, K.floatx())


class TestMultiHead(unittest.TestCase):

    def test_sample(self):
        input_layer = keras.layers.Input(
            shape=(512,),
            name='Input',
        )
        embed_layer = keras.layers.Embedding(
            input_dim=12,
            output_dim=768,
            mask_zero=True,
            name='Embedding',
        )(input_layer)
        output_layer = MultiHeadAttention(
            head_num=12,
            name='Multi-Head',
        )(embed_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()

    def test_invalid_head_num(self):
        with self.assertRaises(IndexError):
            input_layer = keras.layers.Input(
                shape=(2, 3),
                name='Input',
            )
            MultiHeadAttention(
                head_num=2,
                name='Multi-Head',
            )(input_layer)

    def test_fit_self(self):
        input_layer = keras.layers.Input(
            shape=(2, 3),
            name='Input',
        )
        att_layer = MultiHeadAttention(
            head_num=3,
            name='Multi-Head-1',
        )(input_layer)
        dense_layer = keras.layers.Dense(units=3, name='Dense-1')(att_layer)
        att_layer = MultiHeadAttention(
            head_num=3,
            name='Multi-Head-2',
        )(dense_layer)
        output_layer = keras.layers.Dense(units=3, name='Dense-2')(att_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()

        def _generator(batch_size=32):
            while True:
                inputs = np.random.random((batch_size, 2, 3))
                outputs = np.asarray([[[0.0, -0.1, 0.2]] * 2] * batch_size)
                yield inputs, outputs

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=10,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'MultiHeadAttention': MultiHeadAttention,
        })
        for inputs, _ in _generator(batch_size=3):
            predicts = model.predict(inputs)
            expect = np.asarray([[[0.0, -0.1, 0.2]] * 2] * 3)
            actual = np.round(predicts, decimals=1)
            self.assertTrue(np.allclose(expect, actual), (expect, actual))
            break

    def test_fit_multi(self):
        input_query = keras.layers.Input(
            shape=(2, 6),
            name='Input-Q',
        )
        input_key = keras.layers.Input(
            shape=(4, 6),
            name='Input-K',
        )
        input_value = keras.layers.Input(
            shape=(4, 6),
            name='Input-V',
        )
        att_layer = MultiHeadAttention(
            head_num=3,
            name='Multi-Head-1',
        )([input_query, input_key, input_value])
        dense_layer = keras.layers.Dense(units=3, name='Dense-1')(att_layer)
        att_layer = MultiHeadAttention(
            head_num=3,
            name='Multi-Head-2',
        )(dense_layer)
        output_layer = keras.layers.Dense(units=3, name='Dense-2')(att_layer)
        model = keras.models.Model(inputs=[input_query, input_key, input_value], outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()

        def _generator(batch_size=32):
            while True:
                inputs = [
                    np.random.random((batch_size, 2, 6)),
                    np.random.random((batch_size, 4, 6)),
                    np.random.random((batch_size, 4, 6)),
                ]
                outputs = np.asarray([[[0.0, -0.1, 0.2]] * 2] * batch_size)
                yield inputs, outputs

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=10,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            ],
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'MultiHeadAttention': MultiHeadAttention,
        })
        for inputs, _ in _generator(batch_size=3):
            predicts = model.predict(inputs)
            expect = np.asarray([[[0.0, -0.1, 0.2]] * 2] * 3)
            actual = np.round(predicts, decimals=1)
            self.assertTrue(np.allclose(expect, actual), (expect, actual))
            break

    def test_mask_single(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = keras.layers.Embedding(input_dim=3, output_dim=4, mask_zero=True)(input_layer)
        att_layer = MultiHeadAttention(
            head_num=2,
            name='Multi-Head-2',
        )(embed_layer)
        mask_layer = GetMask()(att_layer)
        model = keras.models.Model(inputs=input_layer, outputs=mask_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        predicts = model.predict(np.asarray([[1, 2, 1, 2, 0, 0]])).tolist()
        self.assertEqual([1.0] * 4 + [0.0] * 2, predicts[0], predicts[0])

    def test_mask_multi(self):
        input_q_layer = keras.layers.Input(shape=(None,))
        input_kv_layer = keras.layers.Input(shape=(None,))
        embed_q_layer = keras.layers.Embedding(input_dim=3, output_dim=4, mask_zero=True)(input_q_layer)
        embed_kv_layer = keras.layers.Embedding(input_dim=3, output_dim=4, mask_zero=True)(input_kv_layer)
        att_layer = MultiHeadAttention(
            head_num=2,
            name='Multi-Head-2',
        )([embed_q_layer, embed_kv_layer, embed_kv_layer])
        mask_layer = GetMask()(att_layer)
        model = keras.models.Model(inputs=[input_q_layer, input_kv_layer], outputs=mask_layer)
        model.compile(optimizer='adam', loss='mse', metrics={})
        predicts = model.predict([np.asarray([[1, 2, 1, 2, 0, 0]]), np.asarray([[1, 2, 2, 0, 0, 0]])]).tolist()
        self.assertEqual([1.0] * 4 + [0.0] * 2, predicts[0], predicts[0])

    def test_compare_brute(self):
        for case in range(10):
            batch_size = np.random.randint(1, 10)
            token_num = np.random.randint(2, 10)
            head_num = np.random.randint(1, 5)
            feature_dim = np.random.randint(1, 5) * head_num
            seq_len = np.random.randint(1, 10)
            weights = []
            for i in range(8):
                if i % 2 == 0:
                    weights.append(np.random.random((feature_dim, feature_dim)))
                else:
                    weights.append(np.random.random((feature_dim,)))
            input_q_layer = keras.layers.Input(shape=(None,))
            input_kv_layer = keras.layers.Input(shape=(None,))
            embed_q_layer = keras.layers.Embedding(
                input_dim=token_num,
                output_dim=feature_dim,
                mask_zero=True,
            )(input_q_layer)
            embed_kv_layer = keras.layers.Embedding(
                input_dim=token_num,
                output_dim=feature_dim,
                mask_zero=True,
            )(input_kv_layer)
            att_layer = MultiHeadAttention(
                head_num=head_num,
                weights=weights,
                name='Multi-Head-1',
            )([embed_q_layer, embed_kv_layer, embed_kv_layer])
            att_brute_layer = MultiHeadAttentionBrute(
                head_num=head_num,
                weights=weights,
                name='Multi-Head-2',
            )([embed_q_layer, embed_kv_layer, embed_kv_layer])
            model = keras.models.Model(inputs=[input_q_layer, input_kv_layer], outputs=[att_layer, att_brute_layer])
            model.compile(optimizer='adam', loss='mse', metrics={})
            if case == 0:
                model.summary(line_length=120)
            predicts = model.predict([
                np.random.randint(low=0, high=token_num - 1, size=(batch_size, seq_len)),
                np.random.randint(low=0, high=token_num - 1, size=(batch_size, seq_len)),
            ])
            self.assertTrue(np.allclose(predicts[0], predicts[1]), predicts)
