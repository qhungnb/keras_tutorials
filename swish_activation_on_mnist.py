import os
from time import time

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPool2D, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PARAMS = {
    'filters': (96, 96, 96, 96),
    'kernel_sizes': (3, 3, 3, 2),
    'padding': ('same', 'same', 'same', 'same', 'same'),
    'dense_units': (32, 32),
    'output_dim': 9,
    'input_dim': (28, 28, 1),
    'drop_prob': 0.25,
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 10,
    'save_path': os.path.join(BASE_DIR, 'trained_models/mnist_model_{}.h5'.format(int(time())))
}


def swish(x):
    return x * K.sigmoid(x)


def relu(x, alpha=0.01):
    return K.maximum(x, alpha*x)


activation_func = relu


def build_cnn_model(configs):
    model = Sequential()
    filters = configs.get('filters', (32,))
    kernel_sizes = configs.get('kernel_sizes', (3,))
    padding = configs.get('padding', ('same',))
    dense_units = configs.get('dense_units', (16,))
    output_dim = configs.get('output_dim', 9)
    input_dim = configs.get('input_dim', (28, 28, 1))
    drop_prob = configs.get('drop_prob', 0.25)

    num_cnn_layers = len(filters)
    for i in range(num_cnn_layers):
        if i == 0:
            model.add(Conv2D(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                padding=padding[i],
                input_shape=input_dim
            ))
        else:
            model.add(Conv2D(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                padding=padding[i]
            ))
        model.add(Activation(activation_func))
        model.add(MaxPool2D(pool_size=2, padding='same'))

    model.add(Flatten())

    for dense_unit in dense_units:
        model.add(Dense(
            units=dense_unit
        ))
        model.add(Activation(activation_func))
        model.add(Dropout(drop_prob))

    model.add(Dense(units=output_dim, activation='softmax'))

    model.compile(
        optimizer=configs.get('optimizer', 'adam'),
        loss=configs.get('loss', 'categorical_crossentropy'),
        metrics=configs.get('metrics', ['accuracy'])
    )

    print('Model summary: \n{}'.format(model.summary()))
    return model


def load_dataset():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    configs = MODEL_PARAMS
    x_train, y_train, x_test, y_test = load_dataset()
    configs.update({
        'input_dim': x_train[0].shape,
        'output_dim': y_train.shape[1]
    })

    model = build_cnn_model(configs)

    start_at = time()
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        verbose=2,
        epochs=configs.get('epochs', 20),
        callbacks=[TensorBoard(log_dir='logs/test_swish')]
    )

    end_at = time()
    print('Running time: {}'.format(end_at - start_at))
    model.save(configs.get('save_path'))
