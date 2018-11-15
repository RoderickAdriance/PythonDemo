from __future__ import print_function

import glob
import math
import os
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# mnist_dataframe=pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/mnist_train_small.csv",sep=",",header=None)

mnist_dataframe = pd.read_csv("mnist_data.csv", header=None)

mnist_dataframe = mnist_dataframe.head(10000)
mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))


def parse_labels_and_features(dataset):
    # 目标值,第0列数据  0-9 的数字
    labels = dataset[0]
    # 所有行 , 但是去掉第0列
    features = dataset.loc[:, 1:784]

    features = features / 255.0
    return labels, features

# 处理数据，把特征值缩放到 [0,1]
training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])

validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])

# 显示一个随机样本及其对应的标签。
def showNum():
    rand_example = np.random.choice(training_examples.index)  # rand_example is a index
    _, ax = plt.subplots()
    ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))  # 将图片还原成28X28的图片，就是在对应的像素点填充对应的rgb值
    ax.set_title("Label: %i" % training_targets.loc[rand_example])
    ax.grid(False)
    plt.show()


def construct_feature_columns():
    return {[tf.feature_column.numeric_column('pixels', shape=784)]}


def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
    def _input_fn():
        # 根据 idx 打乱数据顺序
        # idx = np.random.permutation(features.index)
        # raw_features = {"pixels": features.reindex(idx)}
        # raw_targets = np.array(labels[idx])

        #像素并非取 一列,而是所有列都要  [1,2,3,4,5,6.....784]
        raw_features={"pixels":features}
        raw_targets=np.array(labels)

        #组合标签与特征
        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def create_predict_input_fn(features, labels, batch_size):
    def _input_fn():
        raw_features = {"pixels": features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def train_linear_classification_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods
    #测试集
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    #验证集
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    #训练集
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    # 分为 10 个类别
    classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=10,
        optimizer=my_optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1))

    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        classifier.train(input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item["probabilities"] for item in training_predictions])
        training_pred_class_id = np.array([item["class_ids"][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        print(validation_probabilities)
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)

        print("  period %02d : %0.2f" % (period, validation_log_loss))

        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished")

    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier


def train_nn_classification_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    feature_columns = [tf.feature_column.numeric_column('pixels', shape=784)]
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1))

    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []

    for period in range(0,periods):
        classifier.train(input_fn=training_input_fn,steps=steps_per_period)

        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)

        print("  period %02d : %0.2f" % (period, validation_log_loss))

        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier

# classifier = train_linear_classification_model(
#     learning_rate=0.03,
#     steps=1000,
#     batch_size=30,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)

# mnist_test_dataframe = pd.read_csv(
#   "https://dl.google.com/mlcc/mledu-datasets/mnist_test.csv",
#   sep=",",
#   header=None)
#
# test_targets, test_examples = parse_labels_and_features(mnist_test_dataframe)

# mnist_test_dataframe = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/mnist_test.csv",sep=",",header=None)
mnist_test_dataframe=pd.read_csv("mnist_datatest.csv",header=None)
mnist_test_dataframe.to_csv("mnist_datatest.csv",index=False,header=None)
test_targets, test_examples=parse_labels_and_features(mnist_test_dataframe)

classifier = train_nn_classification_model(
    learning_rate=0.04,
    steps=1000,
    batch_size=60,
    hidden_units=[100, 50],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

predict_test_input_fn=create_predict_input_fn(test_examples,test_targets,batch_size=100)

test_predictions=classifier.predict(input_fn=predict_test_input_fn)
test_predictions=np.array([item['class_ids'][0] for item in test_predictions])
accuracy = metrics.accuracy_score(test_targets, test_predictions)
print("Accuracy on test data: %0.2f" % accuracy)