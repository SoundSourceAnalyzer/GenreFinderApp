from __future__ import print_function
import numpy as np
import pickle
import tensorflow as tf


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def one_hot_encoder(self, pos, max):
    encoded = []
    for i in range(0, max):
        if i == pos:
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded


def reformat(self, dataset, labels):
    dataset = np.asarray(dataset).astype(np.float32)
    labels = map(lambda x: np.int32(x), labels)
    labels = map(lambda x: one_hot_encoder(x, 10), labels)
    return dataset, labels


class NeuralNetModel:
    def __init__(self):
        self.hidden_layer1_size = 1024
        self.hidden_layer2_size = 256
        self.hidden_last_layer_size = 56
        self.regularization_meta = 0.03
        self.weights_layer1 = None
        self.biases_layer1 = None
        self.weights_layer2 = None
        self.biases_layer2 = None
        self.weights_layer3 = None
        self.biases_layer3 = None
        self.weights_layer4 = None
        self.biases_layer4 = None
        self.weights = None
        self.biases = None

    def init_weights_and_biases(self, num_features, num_labels):
        self.weights_layer1 = tf.Variable(tf.truncated_normal([num_features, self.hidden_layer1_size], stddev=0.05))
        self.biases_layer1 = tf.Variable(tf.zeros([self.hidden_layer1_size]))

        self.weights_layer2 = tf.Variable(tf.truncated_normal([self.hidden_layer1_size, self.hidden_layer1_size], stddev=0.05))
        self.biases_layer2 = tf.Variable(tf.zeros([self.hidden_layer1_size]))

        self.weights_layer3 = tf.Variable(tf.truncated_normal([self.hidden_layer1_size, self.hidden_layer2_size], stddev=0.05))
        self.biases_layer3 = tf.Variable(tf.zeros([self.hidden_layer2_size]))

        self.weights_layer4 = tf.Variable(tf.truncated_normal([self.hidden_layer2_size, self.hidden_last_layer_size], stddev=0.05))
        self.biases_layer4 = tf.Variable(tf.zeros([self.hidden_last_layer_size]))

        self.weights = tf.Variable(tf.truncated_normal([self.hidden_last_layer_size, num_labels], stddev=0.1))
        self.biases = tf.Variable(tf.zeros([num_labels]))

    def get_nn_4layer(self, dSet, use_dropout):
        input_to_layer1 = tf.matmul(dSet, self.weights_layer1) + self.biases_layer1
        hidden_layer1_output = tf.nn.relu(input_to_layer1)

        if use_dropout:
            dropout_hidden1 = tf.nn.dropout(hidden_layer1_output, self.keep_prob)
            logits_hidden1 = tf.matmul(dropout_hidden1, self.weights_layer2) + self.biases_layer2
        else:
            logits_hidden1 = tf.matmul(hidden_layer1_output, self.weights_layer2) + self.biases_layer2

        hidden_layer2_output = tf.nn.relu(logits_hidden1)

        if use_dropout:
            dropout_hidden2 = tf.nn.dropout(hidden_layer2_output, self.keep_prob)
            logits_hidden2 = tf.matmul(dropout_hidden2, self.weights_layer3) + self.biases_layer3
        else:
            logits_hidden2 = tf.matmul(hidden_layer2_output, self.weights_layer3) + self.biases_layer3

        hidden_layer3_output = tf.nn.relu(logits_hidden2)
        if use_dropout:
            dropout_hidden3 = tf.nn.dropout(hidden_layer3_output, self.keep_prob)
            logits_hidden3 = tf.matmul(dropout_hidden3, self.weights_layer4) + self.biases_layer4
        else:
            logits_hidden3 = tf.matmul(hidden_layer3_output, self.weights_layer4) + self.biases_layer4

        hidden_layer4_output = tf.nn.relu(logits_hidden3)
        if use_dropout:
            dropout_hidden4 = tf.nn.dropout(hidden_layer4_output, self.keep_prob)
            logits = tf.matmul(dropout_hidden4, self.weights) + self.biases
        else:
            logits = tf.matmul(hidden_layer4_output, self.weights) + self.biases

        return logits

    def train_with_gztan(self):
        with open('neuralnet/data/gztan.pickle', 'rb') as f:
            save = pickle.load(f)
            train_dataset_xs = save['train_dataset_xs']
            train_dataset_y = save['train_dataset_y']
            test_dataset_xs = save['test_dataset_xs']
            test_dataset_y = save['test_dataset_y']
            mappings = save['mappings']
            del save  # gc

        num_features = 140
        num_labels = 10
        self.init_weights_and_biases(num_features, num_labels)

        train_dataset, train_labels = self.reformat(train_dataset_xs, train_dataset_y)
        test_dataset, test_labels = self.reformat(test_dataset_xs, test_dataset_y)

        mu = train_dataset.mean(axis=0)
        sigma = train_dataset.std(axis=0)
        train_dataset_scaled = (train_dataset - mu) / sigma
        test_dataset_scaled = (test_dataset - mu) / sigma

        tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
        tf_test_dataset = tf.constant(test_dataset_scaled)

        logits = self.get_nn_4layer(tf_train_dataset, True)
        logits_test = self.get_nn_4layer(tf_test_dataset, False)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.3, global_step, 500, 0.90, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(logits_test)

        num_steps = 1001

        saver = tf.train.Saver()

        with tf.Session() as session:
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in xrange(num_steps):
                offset = (step * self.batch_size) % (len(train_labels) - self.batch_size)
                batch_data = train_dataset_scaled[offset:(offset + self.batch_size)]
                batch_labels = train_labels[offset:(offset + self.batch_size)]

                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, self.keep_prob: 0.7}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            results = test_prediction.eval(feed_dict={self.keep_prob: 1.0})
            print("Test accuracy: %.1f%%" % accuracy(results, test_labels))
            model_path = "neuralnet/model.ckpt"
            save_path = saver.save(session, model_path)
            print("Model saved at " + model_path)
