#!/usr/bin/env python3
import numpy as np
from sys import exit
from os.path import dirname, isdir, exists
from os import walk, makedirs
from scapy.layers.inet import IP
from scapy.utils import PcapReader
from argparse import ArgumentParser
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Input, Concatenate, Dense, BatchNormalization
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
from pickle import load, dump
from df import DFNet
from multiprocessing import Pool

"""
Description:    When this script runs, it will recursively search through directories to find
                PCAP files. For each PCAP file, parse through grabbing timestamps and 
                checking if packet is outgoing or incoming.
                After PCAP is fully parsed, output to a text file (or otherwise specified)
                Continue with the next PCAP until no other PCAP.
                Should combine all the txt files into one directory
Date:           4/26/2021
Usage:          python3 run.py --input-data ./commands
                python3 run.py --load-time ./time_dump.pkl --load-size ./size_dump.pkl --load-classes ./y.pkl
                python3 run.py --load-weights-time ./time_weights/time-model-050-0.708984-0.727088.h5 
                               --load-weights-size ./size_weights/size-model-040-0.720978-0.672098.h5
                python3 run.py --load-time ./time_dump.pkl --load-size ./size_dump.pkl --load-classes ./y.pkl
                               --load-weights-time ./time_weights/time-model-050-0.708984-0.727088.h5 
                               --load-weights-size ./size_weights/size-model-040-0.720978-0.672098.h5
"""

IP_TARGET = ["192.168.128.2", "192.168.1.2", "10.150.101.1", "10.150.101.2", "10.63.1.88", "192.168.1.2", "192.168.128.2", "10.63.1.144"]  # The IP address of the smart home device

# GOOGLE IPs
IP_TARGET = ["10.63.1.88", "10.63.1.24",]
## ALEXA IPs
#IP_TARGET = [""]

SAMPLE_CUTOFF = 800

def save_to_file(sequence, path, delimiter='\t'):
    """save a packet sequence (2-tuple of time and direction) to a file"""
    if not exists(dirname(path)):
        makedirs(dirname(path))
    with open(path, 'w') as file:
        for packet in sequence:
            line = '{t}{b}{d}\n'.format(t=packet[0], b=delimiter, d=packet[1])
            file.write(line)


def preprocessor(inpath, MAX_SIZE):
    """
    :param input: root directory path containing pcap files
    :return: N/A
    """

    print("Processing all pcaps in the " + str(inpath) + " directory...")
    # create list of pcap files to process
    flist = []
    for root, dirs, files in walk(inpath):
        # filter for only pcap files
        files = [fi for fi in files if fi.endswith(".pcap")]
        flist.extend([(root, f) for f in files])

    return feature_extraction(flist, MAX_SIZE)


def task(file):
    #  Open the pcap
    with PcapReader(file[0] + "/" + file[1]) as packets:
        packetCount = 1
        sizeArr = []  # size * direction
        timeArr = []  # time * direction
        iatArr = []  # time * iat
        for packet in packets:
            if packetCount > MAX_SIZE:
                break
            direction = get_packetDirection(packet)
            size = get_packetSize(packet)
            if packetCount == 1:
                startTime = get_packetTime(packet)
                prevTime = startTime
                packetCount += 1
                time = 0
            else:
                endTime = get_packetTime(packet)
                # ensure all the values (size, direction, and time) for each packet exists before adding to numpy array
                if direction != 0 and size != 0 and endTime != 0:
                    time = endTime - startTime
                    time = float(time) * 1000  # Converting to ms
                    iat = float(endTime - prevTime) * 1000
                    prevTime = endTime
                    iatArr.append(iat * direction)
            sizeArr.append(size * direction)
            timeArr.append(time * direction)
        if len(sizeArr) < 50:
            return file, None, None, None, None

        # Padding
        sizeArr.extend([0]*(MAX_SIZE-len(sizeArr)))
        timeArr.extend([0]*(MAX_SIZE-len(timeArr)))
        iatArr.extend([0]*(MAX_SIZE-len(iatArr)))
        if len(sizeArr) > MAX_SIZE:
            sizeArr = sizeArr[:MAX_SIZE]
        if len(timeArr) > MAX_SIZE:
            timeArr = timeArr[:MAX_SIZE]
        if len(iatArr) > MAX_SIZE:
            iatArr = iatArr[:MAX_SIZE]

        numpySizeArr = np.asarray(sizeArr)
        numpyTimeArr = np.asarray(timeArr)
        numpyIatArr = np.asarray(iatArr)

        # construct hand-crafted features
        numpyFeat1Arr = np.asarray([np.mean(np.abs(numpySizeArr)),          # global packet size mean
                                    np.median(np.abs(numpySizeArr)),        # global packet size median
                                    len(numpySizeArr),        # global packet size median
                                    np.mean(numpySizeArr[numpySizeArr > 0]), # total outgoing bytes
                                    np.median(numpySizeArr[numpySizeArr > 0]), # total outgoing bytes
                                    np.sum(numpySizeArr[numpySizeArr > 0]), # total outgoing bytes
                                    np.sum(numpySizeArr[numpyTimeArr > 0]), # total outgoing bytes
                                    len(numpySizeArr[numpySizeArr > 0]),    # total outgoing packets
                                    np.median(numpySizeArr[numpySizeArr < 0]), # total incoming bytes
                                    np.mean(numpySizeArr[numpySizeArr < 0]), # total incoming bytes
                                    np.sum(numpySizeArr[numpySizeArr < 0]), # total incoming bytes
                                    np.sum(numpySizeArr[numpyTimeArr < 0]), # total incoming bytes
                                    len(numpySizeArr[numpySizeArr < 0])     # total incoming packets
                                   ])

        #def window_stack(a, stepsize=1, width=3):
        #    n = a.shape[0]
        #    return np.hstack( list(a[i:1+n+i-width:stepsize] for i in range(0,width)) )
        def window_stack(a, o = 4, w = 2, copy = True):
            sh = (a.size - w + 1, w)
            st = a.strides * 2
            view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
            if copy:
                return view.copy()
            else:
                return view

        INTERVAL_SIZE = 40
        # interval byte counts
        #tmp = np.array_split(np.abs(numpySizeArr), MAX_SIZE//INTERVAL_SIZE)
        tmp = window_stack(np.abs(numpySizeArr), INTERVAL_SIZE//2, INTERVAL_SIZE)
        numpyFeat2Arr = np.mean(tmp, axis=1)
        numpyFeat3Arr = np.median(tmp, axis=1)
        numpyFeat4Arr = np.sum(tmp, axis=1)
        # inflow bytes
        #tmp = np.array(np.array_split(numpySizeArr, MAX_SIZE//INTERVAL_SIZE))
        tmp = window_stack(numpySizeArr, INTERVAL_SIZE//2, INTERVAL_SIZE)
        tmp[tmp > 0] = 0
        numpyFeat5Arr = np.sum(np.abs(tmp), axis=1)
        # outflow bytes
        #tmp = np.array(np.array_split(numpySizeArr, MAX_SIZE//INTERVAL_SIZE))
        tmp = window_stack(numpySizeArr, INTERVAL_SIZE//2, INTERVAL_SIZE)
        tmp[tmp < 0] = 0
        numpyFeat6Arr = np.sum(np.abs(tmp), axis=1)
        # interval center times
        tmp = np.array_split(np.abs(timeArr), MAX_SIZE//INTERVAL_SIZE)
        numpyFeat7Arr = np.median(tmp, axis=1)
        # interval IAT means and total interval
        #tmp = np.array_split(np.abs(iatArr), MAX_SIZE//INTERVAL_SIZE)
        tmp = window_stack(np.abs(iatArr), INTERVAL_SIZE//2, INTERVAL_SIZE)
        numpyFeat8Arr = np.mean(tmp, axis=1)
        numpyFeat9Arr = np.sum(tmp, axis=1)
        # bytes per ms for each interval
        numpyFeat10Arr = np.divide(numpyFeat4Arr, numpyFeat9Arr, out=np.zeros_like(numpyFeat4Arr, dtype=float), where=numpyFeat9Arr!=0)
        numpyFeat11Arr = np.divide(numpyFeat5Arr, numpyFeat9Arr, out=np.zeros_like(numpyFeat5Arr, dtype=float), where=numpyFeat9Arr!=0)
        numpyFeat12Arr = np.divide(numpyFeat6Arr, numpyFeat9Arr, out=np.zeros_like(numpyFeat6Arr, dtype=float), where=numpyFeat9Arr!=0)
        # CUMUL style features
        tmp = np.array_split(numpySizeArr, 10)
        numpyFeat13Arr = np.sum(tmp, axis=1)
        for i in range(1, len(numpyFeat13Arr)):
            numpyFeat13Arr[i] += numpyFeat13Arr[i-1]

        numpyFeatArr = np.concatenate([numpyFeat1Arr,
                                       numpyFeat2Arr,
                                       numpyFeat3Arr,
                                       numpyFeat4Arr,
                                       numpyFeat5Arr,
                                       numpyFeat6Arr,
                                       numpyFeat7Arr,
                                       numpyFeat8Arr,
                                       numpyFeat9Arr,
                                       numpyFeat10Arr,
                                       numpyFeat11Arr,
                                       numpyFeat12Arr,
                                       numpyFeat13Arr,
                                      ])
        numpyFeatArr = np.nan_to_num(numpyFeatArr, nan=0.0, posinf=0.0, neginf=0.0)


        return file, numpySizeArr, numpyTimeArr, numpyIatArr, numpyFeatArr


def feature_extraction(flist, MAX_SIZE):
    """
    :param flist: array of tuples {<path to pcap question directory>, <pcap output file name>}
    :return: numpy2dSizeArray, numpy2dTimeArray
    """
    # initialize two numpy arrays that will hold all the data
    #numpy2dSizeArray = np.empty((0, MAX_SIZE), float)
    #numpy2dTimeArray = np.empty((0, MAX_SIZE), float)
    #numpy2dIatArray = np.empty((0, MAX_SIZE), float)
    #numpy2dFeatArray = None #np.empty((0, N_FEATURES), float)
    numpy2dSizeArray = []
    numpy2dTimeArray = []
    numpy2dIatArray = []
    numpy2dFeatArray = []
    y = list()
    cls_counts = {}
    label_index = 0
    folders_seen = {}
    #  Go through each pcap file = ( path to dir, filename)
    with Pool(os.cpu_count()) as pool:
        for res in pool.imap(task, flist, chunksize=100):
            file, numpySizeArr, numpyTimeArr, numpyIatArr, numpyFeatArr = res
            if numpySizeArr is None:
                continue

            if file[0] not in folders_seen:
                folders_seen[file[0]] = label_index
                print("Loaded class #" + str(label_index) + "...")
                label_index += 1

            if cls_counts.get(folders_seen[file[0]], 0) == SAMPLE_CUTOFF:
                continue
            cls_counts[folders_seen[file[0]]] = 1 + cls_counts.get(folders_seen[file[0]], 0)

            y.append(folders_seen[file[0]])

            numpy2dSizeArray.append(numpySizeArr)
            numpy2dTimeArray.append(numpyTimeArr)
            numpy2dIatArray.append(numpyIatArr)
            numpy2dFeatArray.append(numpyFeatArr)

        numpy2dSizeArray = np.array(numpy2dSizeArray)
        numpy2dTimeArray = np.array(numpy2dTimeArray)
        numpy2dIatArray = np.array(numpy2dIatArray)
        numpy2dFeatArray = np.array(numpy2dFeatArray)

        numpy2dSizeArray, numpy2dTimeArray, numpy2dIatArray, numpy2dFeatArray, y = shuffle(numpy2dSizeArray, numpy2dTimeArray, numpy2dIatArray, numpy2dFeatArray, np.array(y), random_state=100)
        print(cls_counts)
        print(min(cls_counts.values()))
    return numpy2dSizeArray, numpy2dTimeArray, numpy2dIatArray, numpy2dFeatArray, y, label_index


def get_packetDirection(packet):
    try:
        if packet[IP].dst in IP_TARGET:  # packet coming to smart home device
            return -1
        elif packet[IP].src in IP_TARGET:  # packet going from smart home device
            return 1
        return 0  # Error
    except IndexError as e:
        return 0


def get_packetSize(packet):
    try:
        if packet[IP].len is not None:
            return packet[IP].len
        return 0  # Error
    except IndexError as e:
        return 0


def get_packetTime(packet):
    time = packet.time
    return time


def check_args(args):
    if args.input_data is None and args.load_time is None and args.load_size is None and args.load_classes is None:
        exit('Please run the program using either --input-data or --load-time, --load-size, and --load-classes')
    if args.load_weights_time is not None and args.load_weights_size is None:
        exit('Please provide filepaths to weights for both --load-weights-time and --load-weights-size')
    elif args.load_weights_time is None and args.load_weights_size is not None:
        exit('Please provide filepaths to weights for both --load-weights-time and --load-weights-size')


def build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES):
    model = DFNet.build(INPUT_SHAPE, NUM_CLASSES)
    adam = Adam(lr=0.005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


def build_ensemble_model(shape, NUM_CLASSES):
    """
    This ensemble model is based on the Deep Fingerprinting Model's flatten and dense layers
    before classification.
    :param shape: The shape of the ensembled training data
    :return: The ensemble model
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='untruncated_normal')
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Dense(1024, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Activation('relu', name='fc1_act'))
    model.add(Dropout(0.7, name='fc1_dropout'))
    model.add(Dense(1024, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Activation('relu', name='fc2_act'))
    model.add(Dropout(0.7, name='fc2_dropout'))
    model.add(Dense(NUM_CLASSES, kernel_initializer=glorot_uniform(seed=0), name='fc3'))
    model.add(Activation('softmax', name="softmax"))

    sgd = SGD(lr=0.003, nesterov=True, momentum=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def train_SHAME_Model(numpy2dTimeArray, numpy2dSizeArray, numpy2dIatArray, numpy2dFeatArray, INPUT_SHAPE, NUM_CLASSES, time_weights, size_weights, iat_weights, y):
    if not isdir("./time_weights"):
        makedirs("./time_weights")
    if not isdir("./size_weights"):
        makedirs("./size_weights")
    if not isdir("./iat_weights"):
        makedirs("./iat_weights")
    if not isdir("./ensemble_weights"):
        makedirs("./ensemble_weights")
    if not isdir("./pics"):
        makedirs("./pics")

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=100, restore_best_weights=True)

    checkpoint1 = ModelCheckpoint('./time_weights/time-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                  verbose=1, monitor='val_accuracy',
                                  save_best_only=True, mode='auto')
    checkpoint2 = ModelCheckpoint('./size_weights/size-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                  verbose=1, monitor='val_accuracy',
                                  save_best_only=True, mode='auto')
    checkpoint3 = ModelCheckpoint('./iat_weights/iat-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                  verbose=1, monitor='val_accuracy',
                                  save_best_only=True, mode='auto')
    checkpoint4 = ModelCheckpoint('./ensemble_weights/ensemble-model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                                  verbose=1, monitor='val_accuracy',
                                  save_best_only=True, mode='auto')

    tensorboard_callback = TensorBoard(log_dir="./logs")
    # Create two DF models
    model = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)
    model2 = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)
    model3 = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)

    te_cut = int(len(y)*.9)

    #  Check to see if pre-trained weights were passed or not
    batch_size = 256
    if time_weights is None:
        df_time_history = model.fit(numpy2dTimeArray[:te_cut], to_categorical(y[:te_cut]),
                                    batch_size=batch_size,
                                    epochs=600,
                                    validation_split=0.10,
                                    verbose=False,
                                    callbacks=[checkpoint1, early_stopping])
    else:
        model.load_weights(time_weights)

    if size_weights is None:
        df_size_history = model2.fit(numpy2dSizeArray[:te_cut], to_categorical(y[:te_cut]),
                                     batch_size=batch_size,
                                     epochs=600,
                                     validation_split=0.10,
                                     verbose=False,
                                     callbacks=[checkpoint2, early_stopping])
    else:
        model2.load_weights(size_weights)

    if iat_weights is None:
        df_iat_history = model3.fit(numpy2dIatArray[:te_cut], to_categorical(y[:te_cut]),
                                     batch_size=batch_size,
                                     epochs=600,
                                     validation_split=0.10,
                                     verbose=False,
                                     callbacks=[checkpoint3, early_stopping])
    else:  # Load weights
        model3.load_weights(iat_weights)

    #  Make sure to not train either model any further
    model.trainable = False
    model2.trainable = False
    model3.trainable = False
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='./pics/time_model.png', show_shapes='True')
    plot_model(model2, to_file='./pics/size_model2.png', show_shapes='True')
    plot_model(model3, to_file='./pics/iat_model3.png', show_shapes='True')

    print("Getting Flatten layer using the time array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for time
    flatten_model1 = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    outputs1 = flatten_model1.predict(numpy2dTimeArray, verbose=1)  # (N, 1024)

    print("Getting Flatten layer using the size array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model2 = Model(inputs=model2.input, outputs=model2.get_layer('flatten').output)
    outputs2 = flatten_model2.predict(numpy2dSizeArray, verbose=1)  # (N, 1024)

    print("Getting Flatten layer using the iat array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model3 = Model(inputs=model3.input, outputs=model3.get_layer('flatten').output)
    outputs3 = flatten_model3.predict(numpy2dIatArray, verbose=1)  # (N, 1024)

    from sklearn.preprocessing import normalize
    numpy2dFeatArray = normalize(numpy2dFeatArray, norm='l2')
    #print(numpy2dFeatArray[0])

    ensemble_input = np.concatenate((numpy2dFeatArray, outputs1, outputs2, outputs3), axis=1)  # (N, FEATURES+3072)
    print(ensemble_input.shape)
    model4 = build_ensemble_model((ensemble_input.shape[1],), NUM_CLASSES)
    model4.summary()

    #early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model4.fit(x=ensemble_input[:te_cut], y=to_categorical(y[:te_cut]),
               batch_size=256,
               epochs=600,
               validation_split=0.10,
               verbose=False,
               callbacks=[checkpoint4, early_stopping])
    plot_model(model4, to_file='./pics/ensemble_model3.png', show_shapes='True')
    print(model.evaluate(x=numpy2dTimeArray[te_cut:], y=to_categorical(y[te_cut:]), verbose=0))
    print(model2.evaluate(x=numpy2dSizeArray[te_cut:], y=to_categorical(y[te_cut:]), verbose=0))
    print(model3.evaluate(x=numpy2dIatArray[te_cut:], y=to_categorical(y[te_cut:]), verbose=0))
    print(model4.evaluate(x=ensemble_input[te_cut:], y=to_categorical(y[te_cut:]), verbose=0))


def parse_arguments():
    """parse command-line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--input-data", metavar=' ', help='filepath of folders containing pcaps')
    parser.add_argument("--load-time", metavar=' ',
                        help='filepath containing pre-saved data for training time * direction')
    parser.add_argument("--load-size", metavar=' ',
                        help='filepath containing pre-saved data for training size * direction')
    parser.add_argument("--load-iat", metavar=' ',
                        help='filepath containing pre-saved data for training iat * direction')
    parser.add_argument("--load-feat", metavar=' ',
                        help='filepath containing pre-saved data for training hand-crafted features')
    parser.add_argument("--load-classes", metavar=' ', help='filepath containing pre-saved data for training (classes)')
    parser.add_argument("--load-weights-time", metavar=' ', help='filepath to time weights to load into model', default=None)
    parser.add_argument("--load-weights-size", metavar=' ', help='filepath to size weights to load into model', default=None)
    parser.add_argument("--load-weights-iat", metavar=' ', help='filepath to iat weights to load into model', default=None)
    parser.add_argument("--DeepVC", metavar=' ', help='Specifying to use the DeepVC Fingerprinting model instead of the SHAME model')
    args = parser.parse_args()
    check_args(args)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    MAX_SIZE = 1000  # Default
    NUM_CLASSES = None

    if args.input_data is not None:
        # PROCESS DATA
        numpy2dSizeArray, numpy2dTimeArray, numpy2dIatArray, numpy2dFeatArray, y, NUM_CLASSES = preprocessor(args.input_data, MAX_SIZE)
        print("Number of classes " + str(NUM_CLASSES))
        numpy2dSizeArray = numpy2dSizeArray[..., np.newaxis]
        numpy2dTimeArray = numpy2dTimeArray[..., np.newaxis]
        numpy2dIatArray = numpy2dIatArray[..., np.newaxis]
        # SAVE PROCESSED DATA
        with open("size_dump.pkl", "wb") as fp:
            dump(numpy2dSizeArray, fp)
        with open("time_dump.pkl", "wb") as fp:
            dump(numpy2dTimeArray, fp)
        with open("iat_dump.pkl", "wb") as fp:
            dump(numpy2dIatArray, fp)
        with open("feat_dump.pkl", "wb") as fp:
            dump(numpy2dFeatArray, fp)
        with open("y.pkl", "wb") as fp:
            dump(y, fp)
    else:  # not given input data
        if args.load_time is None and args.load_size is None and args.load_classes is None:
            exit('Please provide values for all three arguments (--load-time, --load-size, --load-classes) when loading in your own data')
        else:
            # LOAD PROCESSED DATA
            with open(args.load_time, "rb") as fp:
                numpy2dTimeArray = load(fp)
            with open(args.load_size, "rb") as fp:
                numpy2dSizeArray = load(fp)
            with open(args.load_iat, "rb") as fp:
                numpy2dIatArray = load(fp)
            with open(args.load_feat, "rb") as fp:
                numpy2dFeatArray = load(fp)
            with open(args.load_classes, "rb") as fp:
                y = load(fp)
                NUM_CLASSES = len(np.unique(y))

    INPUT_SHAPE = (MAX_SIZE, 1)

    if args.DeepVC is not None:
        # USE https://arxiv.org/abs/2005.09800
        from DeepVC import CNN
        cnn = CNN()
        modelPath = cnn.train(numpy2dSizeArray, to_categorical(y), NUM_CLASSES)
        modelPath = cnn.train(numpy2dIatArray, to_categorical(y), NUM_CLASSES)
    else:
        # USE Deep Fingerprinting with user weights for the models
        time_weights = args.load_weights_time
        size_weights = args.load_weights_size
        iat_weights = args.load_weights_iat
        train_SHAME_Model(numpy2dTimeArray, numpy2dSizeArray, numpy2dIatArray, numpy2dFeatArray, INPUT_SHAPE, NUM_CLASSES, time_weights, size_weights, iat_weights, y)
