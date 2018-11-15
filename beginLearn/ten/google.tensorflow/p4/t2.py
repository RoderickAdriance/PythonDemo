from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

california_housing_dataframe = pd.read_csv("california_housing.csv")


# 获取特征
def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"]
    )
    return processed_features


# 获取median_house_value
def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    # 取出median_house_value 并缩放median_house_value返回
    output_targets["median_house_value"] = (california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


def splitData():
    # 将数据拆分 训练数据,验证数据
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))

    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

    # correlation_dataframe = training_examples.copy()
    # correlation_dataframe["target"] = training_targets["median_house_value"]
    #
    # print(correlation_dataframe.corr())

    # print("Training examples summary:")
    # display.display(training_examples.describe())
    # print("Validation examples summary:")
    # display.display(validation_examples.describe())
    #
    # print("Training targets summary:")
    # display.display(training_targets.describe())
    # print("Validation targets summary:")
    # display.display(validation_targets.describe())


def construct_feature_columns(input_features):
    # set = set([tf.feature_column.numeric_column(my_feature)
    #            for my_feature in input_features])
    # print(set)
    return set([tf.feature_column.numeric_column(my_feature)
               for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels


def train_model(learning_rate, steps, batch_size, training_examples,
                training_targets, validation_examples, validation_targets
                ):
    periods = 50
    steps_per_period = steps / periods

    # 梯度优化
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    # 创建线性回归模型
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples),
                                                    optimizer=my_optimizer)

    # 创建输入函数
    training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"],
                                            batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
                                                      num_epochs=1,
                                                      shuffle=False)
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # 训练模型
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        # 使用模型对预测数据集进行预测
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        # 获取预测结果
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        print("training_predictions",training_predictions)

        # 使用模型对验证数据集进行验证
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        #获取预测结果
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        print("validation_predictions",validation_predictions)

        # 计算训练和验证的损失   -- 均方根误差
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        print("training_root_mean_squared_error",training_root_mean_squared_error)

        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        #添加均方根误差
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.show()

    return linear_regressor

if __name__ == '__main__':
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))

    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

    minimal_features = [
        "median_income",
        "latitude",
    ]

    minimal_training_examples = training_examples[minimal_features]
    minimal_validation_examples = validation_examples[minimal_features]

    _ = train_model(
        learning_rate=0.01,
        steps=500,
        batch_size=5,
        training_examples=minimal_training_examples,
        training_targets=training_targets,
        validation_examples=minimal_validation_examples,
        validation_targets=validation_targets)

