from __future__ import print_function
import numpy as np
import pickle
import tensorflow as tf
import os


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def one_hot_encoder(pos, max):
    encoded = []
    for i in range(0, max):
        if i == pos:
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded


def randomize(dataset_xs, dataset_y):
    permutation = np.random.permutation(len(dataset_y))
    shuffled_dataset_xs = np.asarray(dataset_xs)[permutation]
    shuffled_dataset_y = np.asarray(dataset_y)[permutation]
    return shuffled_dataset_xs, shuffled_dataset_y


def evens(dataset):
    return dataset[::2]


def odds(dataset):
    return dataset[1::2]


def reformat(dataset, labels, count):
    dataset = np.asarray(dataset).astype(np.float32)
    labels = map(lambda x: np.int32(x), labels)
    labels = map(lambda x: one_hot_encoder(x, count), labels)
    return dataset, labels


class NetworkParams:
    def __init__(self, model_name, num_features, num_labels, train_set, test_set, train_labels, test_labels):
        self.model_name = model_name
        self.num_features = num_features
        self.num_labels = num_labels
        self.train_set = train_set
        self.test_set = test_set
        self.train_labels = train_labels
        self.test_labels = test_labels


class NeuralNetModel:
    def __init__(self):
        self.batch_size = 128
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
        self.keep_prob = None

        self.session = None
        self.saver = None
        self.tf_train_dataset = None
        self.tf_train_labels = None

    def init_weights_and_biases(self, num_features, num_labels):
        self.weights_layer1 = tf.Variable(tf.truncated_normal([num_features, self.hidden_layer1_size], stddev=0.05))
        self.biases_layer1 = tf.Variable(tf.zeros([self.hidden_layer1_size]))

        self.weights_layer2 = tf.Variable(
            tf.truncated_normal([self.hidden_layer1_size, self.hidden_layer1_size], stddev=0.05))
        self.biases_layer2 = tf.Variable(tf.zeros([self.hidden_layer1_size]))

        self.weights_layer3 = tf.Variable(
            tf.truncated_normal([self.hidden_layer1_size, self.hidden_layer2_size], stddev=0.05))
        self.biases_layer3 = tf.Variable(tf.zeros([self.hidden_layer2_size]))

        self.weights_layer4 = tf.Variable(
            tf.truncated_normal([self.hidden_layer2_size, self.hidden_last_layer_size], stddev=0.05))
        self.biases_layer4 = tf.Variable(tf.zeros([self.hidden_last_layer_size]))

        self.weights = tf.Variable(tf.truncated_normal([self.hidden_last_layer_size, num_labels], stddev=0.1))
        self.biases = tf.Variable(tf.zeros([num_labels]))
        self.keep_prob = tf.placeholder(tf.float32)

    def get_nn_4layer(self, data_set, use_dropout):
        input_to_layer1 = tf.matmul(data_set, self.weights_layer1) + self.biases_layer1
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

    def train_gztan_raw(self):

        CURRENT_DIR = os.path.dirname(__file__)
        gztan_raw_path = os.path.join(CURRENT_DIR, 'data/gztan_raw.pickle')
        gztan_vars_path = os.path.join(CURRENT_DIR, 'data/model/gztan_raw_vars.pickle')

        with open(gztan_raw_path, 'rb') as f:
            save = pickle.load(f)
            dataset_xs = save['dataset_xs']
            dataset_y = save['dataset_y']
            del save  # gc

        train_dataset_xs = evens(dataset_xs)
        train_dataset_y = evens(dataset_y)
        test_dataset_xs = odds(dataset_xs)
        test_dataset_y = odds(dataset_y)

        train_dataset_xs, train_dataset_y = randomize(train_dataset_xs, train_dataset_y)
        test_dataset_xs, test_dataset_y = randomize(test_dataset_xs, test_dataset_y)

        train_dataset, train_labels = reformat(train_dataset_xs, train_dataset_y, 10)
        test_dataset, test_labels = reformat(test_dataset_xs, test_dataset_y, 10)

        mu = train_dataset.mean(axis=0)
        sigma = train_dataset.std(axis=0)

        with open(gztan_vars_path, 'wb') as handle:
            pickle.dump([mu, sigma], handle)

        train_dataset_scaled = (train_dataset - mu) / sigma
        test_dataset_scaled = (test_dataset - mu) / sigma

        return self.train(
            NetworkParams("gztan_raw", 145, 10, train_dataset_scaled, test_dataset_scaled, train_labels, test_labels))

    def train(self, network_params):
        self.init_weights_and_biases(network_params.num_features, network_params.num_labels)
        tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, network_params.num_features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, network_params.num_labels))
        tf_test_dataset = tf.constant(network_params.test_set)

        logits = self.get_nn_4layer(tf_train_dataset, True)
        logits_test = self.get_nn_4layer(tf_test_dataset, False)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.3, global_step, 500, 0.90, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(logits_test)

        num_steps = 1001

        saver = tf.train.Saver()

        with tf.Session() as session:
            tf.initialize_all_variables().run()
            print("Session initialized")
            for step in xrange(num_steps):
                offset = (step * self.batch_size) % (len(network_params.train_labels) - self.batch_size)
                batch_data = network_params.train_set[offset:(offset + self.batch_size)]
                batch_labels = network_params.train_labels[offset:(offset + self.batch_size)]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, self.keep_prob: 0.7}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            results = test_prediction.eval(feed_dict={self.keep_prob: 1.0})
            print("Test accuracy: %.1f%%" % accuracy(results, network_params.test_labels))

            model_path = os.path.join(os.path.dirname(__file__),
                                      'data/model/' + network_params.model_name + "_model.ckpt")
            saver.save(session, model_path)
            print("Model saved at " + model_path)

    def predict_gztan(self, feature_array):

        gztan_raw_path = os.path.join(os.path.dirname(__file__), 'data/gztan_raw.pickle')

        with open(gztan_raw_path, 'rb') as f:
            save = pickle.load(f)
            mappings = save['mappings']
            del save  # gc

        with open('/app/neuralnet/data/model/gztan_raw_vars.pickle', 'rb') as handle:
            mu, sigma = pickle.load(handle)

        num_features = 145
        num_labels = 10

        self.init_weights_and_biases(num_features, num_labels)
        self.tf_train_dataset = tf.placeholder(tf.float32, shape=(1, num_features))
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(1, num_labels))

        self.saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(os.path.dirname(__file__), 'data/model/gztan_raw_model.ckpt')

        with tf.Session() as self.session:
            self.saver.restore(self.session, model_path)

            new = np.reshape(feature_array, (1, num_features))
            new = np.asarray(new).astype(np.float32)
            new = (new - mu) / sigma
            label = np.zeros((1, num_labels))

            feed_dict = {self.tf_train_dataset: new, self.tf_train_labels: label, self.keep_prob: 0.7}
            logits = self.get_nn_4layer(self.tf_train_dataset, True)
            predictions = self.session.run(logits, feed_dict=feed_dict)
            genre = (mappings[np.argmax(predictions, 1).item(0)])

        tf.reset_default_graph()
        return genre
