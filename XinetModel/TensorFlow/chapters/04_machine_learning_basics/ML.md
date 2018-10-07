## 通用框架

```{.python .input}
# TF code scaffolding for building simple models.

import tensorflow as tf

# initialize variables/model parameters

# define the training loop operations


def inference(X):
    # compute inference model over data X and return the result
    return


def loss(X, Y):
    # compute loss over training data X and expected values Y
    return


def inputs():
    # read/generate input training data X and expected outputs Y
    return


def train(total_loss):
    # train / adjust model parameters according to computed total loss
    return


def evaluate(sess, X, Y):
    # evaluate the resulting trained model
    return


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
```

## Linear regression

```{.python .input}
# Linear regression example in TF.

import tensorflow as tf

W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    Y_predicted = tf.transpose(inference(X))  # make it a row vector
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))


def inputs():
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [
        63, 28
    ], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23],
        [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
        [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [
        354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254,
        395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244
    ]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    learning_rate = 0.000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(
        total_loss)


def evaluate(sess, X, Y):
    print(sess.run(inference([[50., 20.]])))  # ~ 303
    print(sess.run(inference([[50., 70.]])))  # ~ 256
    print(sess.run(inference([[90., 20.]])))  # ~ 303
    print(sess.run(inference([[90., 70.]])))  # ~ 256


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 10000
    for step in range(training_steps):
        sess.run([train_op])
        if step % 1000 == 0:
            print("Epoch:", step, " loss: ", sess.run(total_loss))

    print("Final model W=", sess.run(W), "b=", sess.run(b))
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
```

## Logistic regression

```{.python .input}
# Logistic regression example in TF using Kaggle's Titanic Dataset.
# Download train.csv from https://www.kaggle.com/c/titanic/data

import tensorflow as tf
import os

# same params and variables initialization as log reg.
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


# former inference is now used for combining inputs
def combine_inputs(X):
    return tf.matmul(X, W) + b


# new inferred value is the sigmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(combine_inputs(X), Y))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer(
        [os.path.join(os.getcwd(), file_name)])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        batch_size=batch_size,
        capacity=batch_size * 50,
        min_after_dequeue=batch_size)


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [
                 ""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

    # convert categorical data
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(
        tf.stack([is_first_class, is_second_class, is_third_class, gender,
                 age]))
    survived = tf.reshape(survived, [100, 1])

    return features, survived


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(
        total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(inference(X) > 0.5, tf.float32)

    print(
        sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    import time
    time.sleep(5)

    coord.request_stop()
    coord.join(threads)
    sess.close()
```

## Softmax

```{.python .input}
# Softmax example in TF using the classical Iris dataset
# Download iris.data from https://archive.ics.uci.edu/ml/datasets/Iris
# Be sure to remove the last empty line of it before running the example

import tensorflow as tf
import os

# this time weights form a matrix, not a column vector, one "weight vector" per class.
W = tf.Variable(tf.zeros([4, 3]), name="weights")
# so do the biases, one per class.
b = tf.Variable(tf.zeros([3]), name="bias")


def combine_inputs(X):
    return tf.matmul(X, W) + b


def inference(X):
    return tf.nn.softmax(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(combine_inputs(X), Y))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer(
        [os.path.dirname(os.path.abspath(__file__)) + "/" + file_name])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        batch_size=batch_size,
        capacity=batch_size * 50,
        min_after_dequeue=batch_size)


def inputs():

    sepal_length, sepal_width, petal_length, petal_width, label =\
        read_csv(100, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])

    # convert class names to a 0 based class index.
    label_number = tf.to_int32(
        tf.argmax(
            tf.to_int32(
                tf.pack([
                    tf.equal(label, ["Iris-setosa"]),
                    tf.equal(label, ["Iris-versicolor"]),
                    tf.equal(label, ["Iris-virginica"])
                ])), 0))

    # Pack all the features that we care about in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(
        tf.pack([sepal_length, sepal_width, petal_length, petal_width]))

    return features, label_number


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(
        total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)

    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
```
