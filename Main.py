# Author: AlexFang, alex.holla@foxmail.com.
import pandas as pd
from captcha_image import ImageCaptcha
import numpy as np
from PIL import Image

import time
import sys
import os
import random

import tensorflow as tf

from Alexnet import Network
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

digits_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

CHAR_NUM = 10


# 1---------------------
def generate_folder():
    if not os.path.exists("./image"):
        os.makedirs("./image")
    for i in os.listdir("./image"):
        if i == "tfrecord":
            continue
        os.remove(os.path.join("./image", i))

    if not os.path.exists("./image/tfrecord"):
        os.makedirs("./image/tfrecord")
    for i in os.listdir("./image/tfrecord"):
        os.remove(os.path.join("./image/tfrecord", i))


def random_captcha_text(char_set, captcha_size):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(idx, num_digits):
    image = ImageCaptcha(width=200, height=50, font_sizes=(40,))
    captcha_text = random_captcha_text(digits_symbols, num_digits)
    captcha_text = "".join(captcha_text)
    image.write(captcha_text, f"./image/{idx:06}_{captcha_text}.png")  # write it


# 2------------------
def generate_train_data(num_samples, num_digits):
    for i in range(num_samples):
        gen_captcha_text_and_image(i, num_digits)
        sys.stdout.write("\r>>creating images %d/%d" % (i + 1, num_samples))
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("All picture has been generated")


# 3--------------------
def save_as_tf(num_test):
    _RANDOM_SEED = 0
    DATASET_DIR = "./image/"
    TFRECORD_DIR = "./image/tfrecord/"

    def _dataset_exists(dataset_dir):
        for split_name in ["train", "test"]:
            output_filename = os.path.join(dataset_dir, split_name + "tfrecords")
            if not tf.io.gfile.exists(output_filename):
                return False
        return True

    def _get_filenames_and_classes(dataset_dir):
        photo_filenames = []
        for filename in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, filename)
            photo_filenames.append(path)
        return photo_filenames

    def int64_feature(values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def image_to_tfexample(image_data, labels):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image": bytes_feature(image_data),
                    "labels": int64_feature(labels),
                }
            )
        )

    def _convert_dataset(split_name, filenames, dataset_dir):
        assert split_name in ["train", "test"]

        with tf.compat.v1.Session() as sess:
            output_filename = os.path.join(TFRECORD_DIR, split_name + ".tfrecords")
            with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i, filename in enumerate(filenames):
                    if "tfrecord" in filename:
                        continue
                    sys.stdout.write(
                        "\r>>changing picture %d / %d" % (i + 1, len(filenames))
                    )
                    sys.stdout.flush()

                    image_data = Image.open(filename)
                    image_data = image_data.resize((224, 224))
                    image_data = np.array(image_data.convert("L"))
                    image_data = image_data.tobytes()

                    labels = os.path.splitext(os.path.basename(filename))[0].split("_")[1]
                    num_labels = list(map(int, labels))

                    example = image_to_tfexample(
                        image_data,
                        num_labels,
                    )
                    tfrecord_writer.write(example.SerializeToString())

            sys.stdout.write("\n")
            sys.stdout.flush()

    if _dataset_exists(DATASET_DIR):
        print("file already exists")
    else:
        photo_filenames = _get_filenames_and_classes(DATASET_DIR)

        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[num_test:]
        testing_filenames = photo_filenames[:num_test]

        _convert_dataset("train", training_filenames, DATASET_DIR)
        _convert_dataset("test", testing_filenames, DATASET_DIR)
        print("-------------We have produced all tfrecord file------------------")


# 4-------------
def train(num_digits, batch_size):
    tf.compat.v1.reset_default_graph()
    TFRECORD_FILE = "./image/tfrecord/train.tfrecords"
    CHECKPOINT_DIR = "./ckpt/"

    # placeholder
    x = tf.compat.v1.placeholder(tf.float32, [None, 224, 224])
    y = tf.compat.v1.placeholder(tf.float32, [None, num_digits])

    lr = tf.Variable(0.001, dtype=tf.float32)

    def read_and_decode(filename):
        filename_queue = tf.compat.v1.train.string_input_producer([filename])
        reader = tf.compat.v1.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                "image": tf.io.FixedLenFeature([], tf.string),
                "labels": tf.io.FixedLenFeature([num_digits], tf.int64),
            },
        )
        image = tf.io.decode_raw(features["image"], tf.uint8)
        image = tf.reshape(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        labels = tf.cast(features["labels"], tf.int32)
        return image, labels

    image, labels = read_and_decode(TFRECORD_FILE)

    (image_batch, label_batch,) = tf.compat.v1.train.shuffle_batch(
        [image, labels],
        batch_size=batch_size,
        capacity=1075,
        min_after_dequeue=1000,
        num_threads=128,
    )

    network = Network(
        num_digits=num_digits,
        num_classes=CHAR_NUM,
        weight_decay=0.0005,
        is_training=True,
    )

    # gpu_options = tf.GPUOptions(allow_growth=True)

    # with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options)) as sess:
    #     gpu_options = tf.GPUOptions(allow_growth=True)
    #     tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options))

    with tf.compat.v1.Session() as sess:

        X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])

        logits, end_pintos = network.construct(X)

        one_hot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=CHAR_NUM)

        total_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.stop_gradient(one_hot_labels), axis=-1
            ),
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(
            total_loss
        )

        correct_prediction = tf.equal(
            tf.argmax(one_hot_labels, 2), tf.argmax(logits, 2)
        )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        # saver.restore(sess, './ckpt/crack_captcha-10000.ckpt')
        # sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10001):
            b_image, b_labels = sess.run([image_batch, label_batch])
            sess.run(
                optimizer,
                feed_dict={
                    x: b_image,
                    y: b_labels,
                },
            )

            if i % 100 == 0:
                if i % 5000 == 0:
                    sess.run(tf.compat.v1.assign(lr, lr / 3))
                acc, loss_ = sess.run(
                    [accuracy, total_loss],
                    feed_dict={
                        x: b_image,
                        y: b_labels,
                    },
                )
                learning_rate = sess.run(lr)
                print(
                    "Iter: %d, Loss:%.3f, Accuracy: %.3f,  Learning_rate:%.7f"
                    % (i, loss_, acc, learning_rate)
                )

                # if acc0 > 0.9 and acc1 > 0.9 and acc2 > 0.9 and acc3 > 0.9 :

                if i % 5000 == 0:
                    # saver.save(sess,'./ckpt/crack_captcha.ckpt', global_step=1)
                    saver.save(
                        sess, CHECKPOINT_DIR + "crack_captcha-" + str(i) + ".ckpt"
                    )
                    print("Save model %s------" % str(i))
                    continue
        coord.request_stop()
        coord.join(threads)


# 5-------------
def test(num_digits):
    batch_size = 1
    TFRECORD_FILE = "./image/tfrecord/test.tfrecords"

    x = tf.compat.v1.placeholder(tf.float32, [None, 224, 224])

    def read_and_decode(filename):
        filename_queue = tf.compat.v1.train.string_input_producer([filename])
        reader = tf.compat.v1.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                "image": tf.io.FixedLenFeature([], tf.string),
                "labels": tf.io.FixedLenFeature([num_digits], tf.int64),
            },
        )
        image = tf.io.decode_raw(features["image"], tf.uint8)
        image_raw = tf.reshape(image, [224, 224])  # raw data

        image = tf.reshape(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0  # standardlize
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        labels = tf.cast(features["labels"], tf.int32)
        return image, image_raw, labels

    # get label
    image, image_raw, labels = read_and_decode(TFRECORD_FILE)
    # print(len(sess.run(image)))
    (image_batch, image_raw_batch, label_batch,) = tf.compat.v1.train.shuffle_batch(
        [image, image_raw, labels],
        batch_size=batch_size,
        capacity=53,
        min_after_dequeue=50,
        num_threads=1,
    )

    network = Network(num_classes=CHAR_NUM, weight_decay=0.0005, is_training=True)
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    # with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options)) as sess:
    with tf.compat.v1.Session() as sess:
        X = tf.reshape(x, [batch_size, 224, 224, 1])

        logits, end_pintos = network.construct(X)

        predictions = tf.reshape(logits, [-1, num_digits, CHAR_NUM])
        predictions = tf.argmax(predictions, -1)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "./ckpt/crack_captcha-10000.ckpt")

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(5):
            b_image, b_image_raw, b_labels = sess.run(
                [
                    image_batch,
                    image_raw_batch,
                    label_batch,
                ]
            )

            # img = np.array(b_image_raw[0],dtype=np.uint8)

            # [1,224,224]
            img = Image.fromarray(b_image_raw[0], "L")
            """
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            """
            print("label:", b_labels)

            labels = sess.run(
                [prediction0, prediction1, prediction2, prediction3],
                feed_dict={x: b_image},
            )
            print("predict:", labels)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    NUM_DIGITS = 4
    NUM_SAMPLES = 1000
    NUM_TEST = 50
    BATCH_SIZE = 16

    generate_folder()
    generate_train_data(num_samples=NUM_SAMPLES, num_digits=NUM_DIGITS)
    save_as_tf(num_test=NUM_TEST)
    train(num_digits=NUM_DIGITS, batch_size=BATCH_SIZE)
    test(num_digits=NUM_DIGITS)
