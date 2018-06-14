import os
import tensorflow as tf

def save_model(sess, model_filename):
    """
    :param sess:
    :param filename:
    :return:
    """
    if not os.path.exists(MODEL_FOLDER):
        print('Creating path where to save model: ' + MODEL_FOLDER)

        os.mkdir(MODEL_FOLDER)

    print('Saving model at: ' + model_filename)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.save(sess, MODEL_FOLDER + model_filename)
    print('Model successfully saved.\n')


def load_model(sess, filename):
    """
    :param sess:
    :param filename:
    :return:
    """
    if os.path.exists(filename):
        print('\nLoading saved model from: ' + filename)
        saver = tf.train.Saver()
        saver.restore(sess, filename)
        print('Model successfully loaded.\n')
        return True
    else:
        print('Model file <<' + filename + '>> does not exists!')
        return False
