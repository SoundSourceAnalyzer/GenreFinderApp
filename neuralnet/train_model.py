from __future__ import print_function
import numpy as np
import pickle
import tensorflow as tf


def one_hot_encoder(pos, max):
    encoded = []
    for i in range(0, max):
        if i == pos:
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded


def reformat(dataset, labels):
    dataset = np.asarray(dataset).astype(np.float32)
    labels = map(lambda x: np.int32(x), labels)
    labels = map(lambda x: one_hot_encoder(x, 10), labels)
    return dataset, labels


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def get_nn_4layer(dSet, use_dropout):
    input_to_layer1 = tf.matmul(dSet, weights_layer1) + biases_layer1
    hidden_layer1_output = tf.nn.relu(input_to_layer1)

    logits_hidden1 = None
    if use_dropout:
        dropout_hidden1 = tf.nn.dropout(hidden_layer1_output, keep_prob)
        logits_hidden1 = tf.matmul(dropout_hidden1, weights_layer2) + biases_layer2
    else:
        logits_hidden1 = tf.matmul(hidden_layer1_output, weights_layer2) + biases_layer2

    hidden_layer2_output = tf.nn.relu(logits_hidden1)

    logits_hidden2 = None
    if use_dropout:
        dropout_hidden2 = tf.nn.dropout(hidden_layer2_output, keep_prob)
        logits_hidden2 = tf.matmul(dropout_hidden2, weights_layer3) + biases_layer3
    else:
        logits_hidden2 = tf.matmul(hidden_layer2_output, weights_layer3) + biases_layer3

    hidden_layer3_output = tf.nn.relu(logits_hidden2)
    logits_hidden3 = None
    if use_dropout:
        dropout_hidden3 = tf.nn.dropout(hidden_layer3_output, keep_prob)
        logits_hidden3 = tf.matmul(dropout_hidden3, weights_layer4) + biases_layer4
    else:
        logits_hidden3 = tf.matmul(hidden_layer3_output, weights_layer4) + biases_layer4

    hidden_layer4_output = tf.nn.relu(logits_hidden3)
    logits = None
    if use_dropout:
        dropout_hidden4 = tf.nn.dropout(hidden_layer4_output, keep_prob)
        logits = tf.matmul(dropout_hidden4, weights) + biases
    else:
        logits = tf.matmul(hidden_layer4_output, weights) + biases

    return logits


with open('neuralnet/data/gztan.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset_xs = save['train_dataset_xs']
    train_dataset_y = save['train_dataset_y']
    test_dataset_xs = save['test_dataset_xs']
    test_dataset_y = save['test_dataset_y']
    mappings = save['mappings']
    del save  # gc

train_dataset, train_labels = reformat(train_dataset_xs, train_dataset_y)
test_dataset, test_labels = reformat(test_dataset_xs, test_dataset_y)

mu = train_dataset.mean(axis=0)
sigma = train_dataset.std(axis=0)
train_dataset_scaled = (train_dataset - mu) / sigma
test_dataset_scaled = (test_dataset - mu) / sigma

batch_size = 128
num_labels = 10
num_features = 140
hidden_layer1_size = 1024
hidden_layer2_size = 256
hidden_lastlayer_size = 56

regularization_meta = 0.03

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset_scaled)
    keep_prob = tf.placeholder(tf.float32)

    weights_layer1 = tf.Variable(tf.truncated_normal([num_features, hidden_layer1_size], stddev=0.05))
    biases_layer1 = tf.Variable(tf.zeros([hidden_layer1_size]))

    weights_layer2 = tf.Variable(
        tf.truncated_normal([hidden_layer1_size, hidden_layer1_size], stddev=0.05))
    biases_layer2 = tf.Variable(tf.zeros([hidden_layer1_size]))

    weights_layer3 = tf.Variable(
        tf.truncated_normal([hidden_layer1_size, hidden_layer2_size], stddev=0.05))
    biases_layer3 = tf.Variable(tf.zeros([hidden_layer2_size]))

    weights_layer4 = tf.Variable(
        tf.truncated_normal([hidden_layer2_size, hidden_lastlayer_size], stddev=0.05))
    biases_layer4 = tf.Variable(tf.zeros([hidden_lastlayer_size]))

    weights = tf.Variable(tf.truncated_normal([hidden_lastlayer_size, num_labels], stddev=0.1))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = get_nn_4layer(tf_train_dataset, True)
    logits_test = get_nn_4layer(tf_test_dataset, False)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.3, global_step, 500, 0.90, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(logits_test)

    num_steps = 1001

    saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in xrange(num_steps):
            offset = (step * batch_size) % (len(train_labels) - batch_size)
            batch_data = train_dataset_scaled[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.7}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        results = test_prediction.eval(feed_dict={keep_prob: 1.0})
        print("Test accuracy: %.1f%%" % accuracy(results, test_labels))
        model_path = "neuralnet/model.ckpt"
        save_path = saver.save(session, model_path)
        print("Model saved at " + model_path)