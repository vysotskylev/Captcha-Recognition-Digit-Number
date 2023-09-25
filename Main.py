# Author: AlexFang, alex.holla@foxmail.com.
import pandas as pd
from captcha_image import ImageCaptcha
import numpy as np
from PIL import Image

import time
import sys
import os
import random
import re

import tensorflow as tf

from Alexnet import Network
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

digits_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

CHECKPOINT_PREFIX = "crack_captcha_"

CHAR_NUM = 10
IMAGE_CHUNK_SIZE = 1_000
LEARNING_RATE_DROP_ITERS = 20_000
CHECKPOINT_ITERS = 10_000
STAT_PRINT_ITERS = 200
TOTAL_ITERS = 100_000 + 1


# 1---------------------
def generate_folder(data_dir):
    image_dir = os.path.join(data_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    for chunk_dir in os.listdir(image_dir):
        for f in os.listdir(os.path.join(image_dir, chunk_dir)):
            os.remove(os.path.join(image_dir, chunk_dir, f))
    tfrecord_dir = os.path.join(data_dir, "tfrecord")
    os.makedirs(tfrecord_dir, exist_ok=True)
    for i in os.listdir(tfrecord_dir):
        os.remove(os.path.join(tfrecord_dir, i))


def random_captcha_text(char_set, captcha_size):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(idx, num_digits, data_dir, chunk_size):
    image = ImageCaptcha(width=200, height=50, font_sizes=(40,))
    captcha_text = random_captcha_text(digits_symbols, num_digits)
    captcha_text = "".join(captcha_text)
    subdir = f"{idx - idx % chunk_size:06}"
    os.makedirs(os.path.join(data_dir, "image", subdir), exist_ok=True)
    image.write(captcha_text, os.path.join(data_dir, "image", subdir, f"{idx:06}_{captcha_text}.png"))  # write it


# 2------------------
def generate_train_data(num_samples, num_digits, data_dir):
    for i in range(num_samples):
        gen_captcha_text_and_image(i, num_digits, data_dir, IMAGE_CHUNK_SIZE)
        sys.stdout.write("\r>>creating images %d/%d" % (i + 1, num_samples))
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("All pictures has been generated")


# 3--------------------
def save_as_tf(num_test, data_dir):
    _RANDOM_SEED = 0
    DATASET_DIR = os.path.join(data_dir, "image")
    VALIDATION_DIR = os.path.join(data_dir, "validation")
    TFRECORD_DIR = os.path.join(data_dir, "tfrecord")

    def _dataset_exists(dataset_dir):
        for split_name in ["train", "test", "validation"]:
            output_filename = os.path.join(dataset_dir, split_name + ".tfrecords")
            if not tf.io.gfile.exists(output_filename):
                return False
        return True

    def _get_filenames_and_classes(dataset_dir):
        photo_filenames = []
        for chunk_dir in os.listdir(dataset_dir):
            for filename in os.listdir(os.path.join(dataset_dir, chunk_dir)):
                path = os.path.join(dataset_dir, chunk_dir, filename)
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

    def _convert_dataset(split_name, filenames):
        assert split_name in ["train", "test", "validation"]

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

    if _dataset_exists(TFRECORD_DIR):
        print("file already exists")
    else:
        photo_filenames = _get_filenames_and_classes(DATASET_DIR)
        validation_filenames = _get_filenames_and_classes(VALIDATION_DIR)

        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[num_test:]
        testing_filenames = photo_filenames[:num_test]


        _convert_dataset("train", training_filenames)
        _convert_dataset("test", testing_filenames)
        _convert_dataset("validation", validation_filenames)
        print("-------------We have produced all tfrecord file------------------")


def read_and_decode(filename, num_digits):
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


# 4-------------
def train(num_digits, batch_size, data_dir, restore_checkpoint=False, flush_callback=None):
    tf.compat.v1.reset_default_graph()

    DATASET_DIR = os.path.join(data_dir, "image")
    TFRECORD_DIR = os.path.join(data_dir, "tfrecord")

    train_record = os.path.join(TFRECORD_DIR, "train.tfrecords")
    test_record = os.path.join(TFRECORD_DIR, "test.tfrecords")
    val_record = os.path.join(TFRECORD_DIR, "validation.tfrecords")

    CHECKPOINT_DIR = os.path.join(data_dir, "ckpt")

    # placeholder
    x = tf.compat.v1.placeholder(tf.float32, [None, 224, 224])
    y = tf.compat.v1.placeholder(tf.float32, [None, num_digits])

    lr = tf.Variable(0.0003, dtype=tf.float32)

    train_images, _, train_labels = read_and_decode(train_record, num_digits)
    test_images, _, test_labels = read_and_decode(train_record, num_digits)
    val_images, _, val_labels = read_and_decode(train_record, num_digits)

    (train_image_batch, train_label_batch,) = tf.compat.v1.train.shuffle_batch(
        [train_images, train_labels],
        batch_size=batch_size,
        capacity=1075,
        min_after_dequeue=1000,
        num_threads=128,
    )

    (test_image_batch, test_label_batch,) = tf.compat.v1.train.shuffle_batch(
        [test_images, test_labels],
        batch_size=batch_size,
        capacity=1075,
        min_after_dequeue=1000,
        num_threads=128,
    )

    (val_image_batch, val_label_batch,) = tf.compat.v1.train.shuffle_batch(
        [val_images, val_labels],
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

    with tf.compat.v1.Session() as sess:

        X = tf.reshape(x, [batch_size, 224, 224, 1])

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

        if restore_checkpoint:
            ckpt = get_most_recent_checkpoint(data_dir)
            if ckpt is None:
                print("WARNING: No checkpoint found!")
            else:
                print(f"Restoring from {ckpt}")
                saver.restore(sess, ckpt)

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(TOTAL_ITERS):
            b_train_images, b_train_labels = sess.run([train_image_batch, train_label_batch])
            _, train_loss = sess.run(
                [optimizer, total_loss],
                feed_dict={
                    x: b_train_images,
                    y: b_train_labels,
                },
            )

            if i % STAT_PRINT_ITERS == 0:
                if i % LEARNING_RATE_DROP_ITERS == 0 and i > 0:
                    sess.run(tf.compat.v1.assign(lr, lr / 3))
                b_test_images, b_test_labels = sess.run([test_image_batch, test_label_batch])
                test_acc, = sess.run(
                    [accuracy],
                    feed_dict={
                        x: b_test_images,
                        y: b_test_labels,
                    },
                )

                b_val_images, b_val_labels = sess.run([val_image_batch, val_label_batch])
                val_acc, = sess.run(
                    [accuracy],
                    feed_dict={
                        x: b_val_images,
                        y: b_val_labels,
                    },
                )
                learning_rate = sess.run(lr)
                print(
                    f"Iter: {i}, Train loss: {train_loss:.3}, Test acc: {test_acc:.3}, Val acc: {val_acc:.3}, LR: {learning_rate:.7}"
                )

                if i % CHECKPOINT_ITERS == 0 and i > 0:
                    saver.save(
                        sess, os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}{i}.ckpt")
                    )
                    if flush_callback is not None:
                        flush_callback()
                    print("Save model %s------" % str(i))
                    continue
        coord.request_stop()
        coord.join(threads)


def find_oldest_file_in_directory(directory, pattern):
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return None

    # Initialize variables to track the oldest file and its timestamp
    oldest_file = None
    oldest_timestamp = float('inf')  # Start with a high value

    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Traverse only the files in the directory (excluding subdirectories)
    for filename in os.listdir(directory):
        if regex.match(filename):
            file_path = os.path.join(directory, filename)
            file_timestamp = os.path.getmtime(file_path)

            # Update the oldest file if needed
            if file_timestamp < oldest_timestamp:
                oldest_file = file_path
                oldest_timestamp = file_timestamp

    return oldest_file


def get_most_recent_checkpoint(data_dir):
    CHECKPOINT_DIR = os.path.join(data_dir, "ckpt")
    path = find_oldest_file_in_directory(CHECKPOINT_DIR, CHECKPOINT_PREFIX + r"\d+\.ckpt\..+")
    if path is None:
        return None
    return ".".join(path.split(".")[:-1])


# 5-------------
def test(num_digits, data_dir):
    batch_size = 1
    
    DATASET_DIR = os.path.join(data_dir, "image")
    TFRECORD_DIR = os.path.join(data_dir, "tfrecord")
    CHECKPOINT_DIR = os.path.join(data_dir, "ckpt")

    TFRECORD_FILE = os.path.join(TFRECORD_DIR, "test.tfrecords")

    x = tf.compat.v1.placeholder(tf.float32, [None, 224, 224])

    # get label
    image, image_raw, labels = read_and_decode(TFRECORD_FILE, num_digits)
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
        saver.restore(sess, get_most_recent_checkpoint(data_dir))

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
    NUM_DIGITS = 6
    NUM_SAMPLES = 1000
    NUM_TEST = 50
    BATCH_SIZE = 16
    DATA_DIR = "."

    generate_folder(data_dir=DATA_DIR)
    generate_train_data(num_samples=NUM_SAMPLES, num_digits=NUM_DIGITS, data_dir=DATA_DIR)
    save_as_tf(num_test=NUM_TEST, data_dir=DATA_DIR)
    train(num_digits=NUM_DIGITS, batch_size=BATCH_SIZE, data_dir=DATA_DIR, restore_checkpoint=True)
    test(num_digits=NUM_DIGITS, data_dir=DATA_DIR)
