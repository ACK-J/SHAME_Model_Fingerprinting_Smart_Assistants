import os

if '1' == os.getenv('useGpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Dropout, MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Sequential
from keras.callbacks import TensorBoard
import keras.backend as K

modelDir = 'modelDir'
if not os.path.isdir(modelDir):
    os.makedirs(modelDir)


class CNN():
    def __init__(self, name='cnn'):
        self.params = {
            'optimizer': "Adam",
            'learning_rate': 0.05,
            'activation1': "tanh",
            'activation2': "selu",
            'activation3': "elu",
            'activation4': "selu",
            'drop_rate1': 0.2,
            'drop_rate2': 0.1,
            'drop_rate3': 0.4,
            'drop_rate4': 0.5,
            'decay': 0.5,
            'batch_size': 150,
            'data_dim': 1000,
            'epochs': 100,
            'conv1': 256,
            'conv2': 32,
            'conv3': 128,
            'conv4': 32,
            'pool1': 3,
            'pool2': 2,
            'pool3': 1,
            'pool4': 2,
            'kernel_size1': 9,
            'kernel_size2': 9,
            'kernel_size3': 11,
            'kernel_size4': 15,
            'dense1': 120,
            'dense2': 140,
            'dense1_act': "selu",
            'dense2_act': "selu"
        }
        self.verbose = True
        self.plotModel = True
        self.name = name

    def create_model(self, NUM_CLASS):
        print('Creating model...')

        layers = [
            Conv1D(self.params['conv1'], kernel_size=self.params['kernel_size1'], activation=self.params['activation1'],
                   input_shape=(self.params['data_dim'], 1), use_bias=False, kernel_initializer='glorot_normal'),
            BatchNormalization(),
            MaxPooling1D(self.params['pool1']),
            Dropout(rate=self.params['drop_rate1']),

            Conv1D(self.params['conv2'], kernel_size=self.params['kernel_size2'], activation=self.params['activation2'],
                   kernel_initializer='glorot_normal'),
            BatchNormalization(),
            MaxPooling1D(self.params['pool2']),
            Dropout(rate=self.params['drop_rate2']),

            Conv1D(self.params['conv3'], kernel_size=self.params['kernel_size3'], activation=self.params['activation3'],
                   kernel_initializer='glorot_normal'),
            BatchNormalization(),
            MaxPooling1D(self.params['pool3']),
            Dropout(rate=self.params['drop_rate3']),

            Conv1D(self.params['conv4'], kernel_size=self.params['kernel_size4'], activation=self.params['activation4'],
                   kernel_initializer='glorot_normal'),
            BatchNormalization(),
            MaxPooling1D(self.params['pool4']),
            GlobalAveragePooling1D(),

            Dense(self.params['dense1'], activation=self.params['dense1_act'], kernel_initializer='glorot_normal'),
            BatchNormalization(),
            Dense(self.params['dense2'], activation=self.params['dense2_act'], kernel_initializer='glorot_normal'),
            BatchNormalization(),
            Dense(NUM_CLASS, activation='softmax')]

        model = Sequential(layers)

        print('Compiling...')
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.params['optimizer'],
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train, NUM_CLASS):
        '''train the cnn model'''
        model = self.create_model(NUM_CLASS)
        if self.plotModel:
            picDir = os.path.join(modelDir, 'pic')
            if not os.path.isdir(picDir):
                os.makedirs(picDir)
            picPath = os.path.join(picDir, 'cnn_model.png')
            from keras.utils import plot_model
            plot_model(model, to_file=picPath, show_shapes='True')

        print('Fitting model...')

        def lr_scheduler(epoch):
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * self.params['decay'])
                print("lr changed to {}".format(lr * self.params['decay']))
            return K.get_value(model.optimizer.lr)

        modelPath = os.path.join(modelDir, 'cnn_weights_best.hdf5')
        # checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        #
        # CallBacks = [checkpointer]
        # if self.params['optimizer'] == 'SGD':
        #     scheduler = LearningRateScheduler(lr_scheduler)
        #     CallBacks.append(scheduler)
        # CallBacks.append(EarlyStopping(monitor='val_accuracy', mode='max', patience=6))
        # CallBacks.append(TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True))
        tensorboard_callback = TensorBoard(log_dir="./logs")
        hist = model.fit(X_train, y_train,
                         batch_size=self.params['batch_size'],
                         epochs=self.params['epochs'],
                         validation_split=0.2,
                         verbose=self.verbose,
                         callbacks=[tensorboard_callback])

        if self.plotModel:
            from keras.utils import plot_model
            plot_model(model, to_file='model.png', show_shapes='True')

        return modelPath

    def prediction(self, X_test, NUM_CLASS, modelPath):
        print('Predicting results with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        y_pred = model.predict(X_test)
        return y_pred

    def test(self, X_test, y_test, NUM_CLASS, modelPath):
        print('Testing with best model...')
        model = self.create_model(NUM_CLASS)
        model.load_weights(modelPath)
        score, acc = model.evaluate(X_test, y_test, batch_size=100)

        print('Test score:', score)
        print('Test accuracy:', acc)
        return acc
