import tensorflow as tf
import pickle
import numpy as np

batch_size = 128
num_labels = 10
num_features = 140
hidden_layer1_size = 1024
hidden_layer2_size = 256
hidden_lastlayer_size = 56

regularization_meta = 0.03


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


with open('neuralnet/data/gztan.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset_xs = save['train_dataset_xs']
    train_dataset_y = save['train_dataset_y']
    test_dataset_xs = save['test_dataset_xs']
    test_dataset_y = save['test_dataset_y']
    mappings = save['mappings']
    del save  # gc


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

train_dataset, train_labels = reformat(train_dataset_xs, train_dataset_y)
test_dataset, test_labels = reformat(test_dataset_xs, test_dataset_y)

mu = train_dataset.mean(axis=0)
sigma = train_dataset.std(axis=0)
train_dataset_scaled = (train_dataset - mu) / sigma
test_dataset_scaled = (test_dataset - mu) / sigma

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

saver = tf.train.Saver()
model_path = "neuralnet/model.ckpt"

with tf.Session() as session:
    saver.restore(session, model_path)
    print("Model restored")
    # Here comes the magic
    new = np.reshape(train_dataset_scaled[0], (1,140))
    label = np.zeros((1, 10))
    tf_train_dataset = tf.placeholder(tf.float32, shape=(1, 140))
    tf_train_labels = tf.placeholder(tf.float32, shape=(1, 10))
    keep_prob = tf.placeholder(tf.float32)
    feed_dict = {tf_train_dataset: new, tf_train_labels: label, keep_prob: 0.7}
    logits = get_nn_4layer(tf_train_dataset, True)
    print(session.run(logits, feed_dict=feed_dict))