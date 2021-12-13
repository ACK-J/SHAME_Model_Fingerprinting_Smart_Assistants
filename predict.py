import numpy as np
from tensorflow.keras.models import Model, Sequential
import time

from run import build_deep_fingerprinting_model, build_ensemble_model, preprocessor, feature_extraction

if __name__ == '__main__':
    pcap = "./predict"
    MAX_SIZE = 1000
    numpy2dSizeArray, numpy2dTimeArray, _, NUM_CLASSES = preprocessor(pcap, MAX_SIZE)
    numpy2dSizeArray = numpy2dSizeArray[..., np.newaxis]
    numpy2dTimeArray = numpy2dTimeArray[..., np.newaxis]

    INPUT_SHAPE = (MAX_SIZE, 1)
    NUM_CLASSES = 20
    # Create two DF models
    model = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)
    model2 = build_deep_fingerprinting_model(INPUT_SHAPE, NUM_CLASSES)


    #model.load_weights("./Google/time_weights/time-model-057-0.996356-0.994062.h5")
    #model2.load_weights("./Google/size_weights/size-model-068-0.997594-0.997990.h5")

    # model.load_weights("./alexa/time_weights/time-model-041-0.886523-0.864234.h5")
    # model2.load_weights("./alexa/size_weights/size-model-035-0.932176-0.919757.h5")

    model.load_weights("./20_question_alexa/time-model-057-0.862883-0.861824.h5")
    model2.load_weights("./20_question_alexa/size-model-067-0.902822-0.862916.h5")

    #  Make sure to not train either model any further
    model.trainable = False
    model2.trainable = False

    START = time.time()
    print("Getting Flatten layer using the time array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for time
    flatten_model1 = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    outputs1 = flatten_model1.predict(numpy2dTimeArray, verbose=1)  # (N, 1024)

    print("Getting Flatten layer using the size array")
    #  Create a new model that takes in (MAX_SIZE, 1) and outputs the flatten layers for size
    flatten_model2 = Model(inputs=model2.input, outputs=model2.get_layer('flatten').output)
    outputs2 = flatten_model2.predict(numpy2dSizeArray, verbose=1)  # (N, 1024)

    #  Combine the two models outputs, just created
    ensemble_input = np.concatenate((outputs1, outputs2), axis=1)  # (N, 2048) (samples, combined flattened layer)
    ensemble_model = build_ensemble_model(ensemble_input.shape, NUM_CLASSES)

    #ensemble_model.load_weights("./Google/ensemble_weights/ensemble-model-096-0.999033-0.999208.h5")
    #ensemble_model.load_weights("./alexa/ensemble_weights/ensemble-model-001-0.951374-0.956835.h5")
    ensemble_model.load_weights("./20_question_alexa/ensemble-model-002-0.960283-0.927167.h5")


    predictions = ensemble_model.predict(ensemble_input)
    print(predictions)
    STOP = time.time()
    classes = np.argmax(predictions, axis=1)
    print(classes)

    print(STOP - START)
