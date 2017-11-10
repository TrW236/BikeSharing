import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def make_dummy(data, dummy_fields=None):
    if dummy_fields is None:
        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday'];
    for each in dummy_fields:
        dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)
        data = pd.concat([data, dummies], axis=1)
    return data


def drop_fields(data, fields_to_drop=None):
    if fields_to_drop is None:
        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr'];
    data_res = data.drop(fields_to_drop, axis=1)
    return data_res


# Store scalings in a dictionary so we can convert back later
def scale_features(data, features=None):
    if features is None:
        features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    scaled_features = {}
    for each in features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean) / std
    return scaled_features


def split_data(data, target_fields=None):
    # Save data for approximately the last 21 days
    if target_fields is None:
        target_fields = ['cnt', 'casual', 'registered']
    test_data = data[-21 * 24:]
    # Now remove the test data from the data set
    data = data[:-21 * 24]
    # Separate the data into features and targets

    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
    # Hold out the last 60 days or so of the remaining data as a validation set
    train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
    val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]
    return train_features, train_targets, val_features, val_targets, test_features, test_targets


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # sigmoid function
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))
        # def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        # self.activation_function = sigmoid

    def relu(self, X):  # X.shape = (n, 3)
        ret = []
        for x in X:
            if x >= 0:
                ret.append(x)
            else:
                ret.append(0.0)
        return np.array(ret)

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.
            Arguments
            ---------
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):  # X(n_i,)  y(1,)
            ### Forward pass ###
            # Hidden layer
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)  # signals into hidden layer
            # (n_i,) np.dot (n_i, n_h) = (n_h,)
            hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
            # (n_h) = (n_h)

            # Output layer
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
            # (n_h) np.dot (n_h , 1) = (1,)
            final_outputs = final_inputs  # self.acti_func_out(final_inputs) # signals from final output layer
            # (1,)

            ### Backward pass ###
            # Output error
            error = y - final_outputs  # Output layer error is the difference between desired target and actual output.
            # (1,)
            # Backpropagated error terms
            output_error_term = error  # f'(x) = 1 derivative of activation func
            # (1,) * (1,) = (1,)

            # Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
            # (n_h, 1) np.dot (1,) = (n_h,)
            hidden_error_term = hidden_error * (hidden_outputs * (1 - hidden_outputs))
            # (n_h,) * (n_h,) = (n_h, )

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:, None]  # X[:,None] make shape changed from (n_i,) to  (n_i, 1)
            # (n_h,) * (n_i, 1) = (n_i, n_h)
            # Weight step (hidden to output)
            delta_weights_h_o += (output_error_term * hidden_outputs)[:, None]
            # (1,) * (n_h,) = (n_h,) -> (n_h, 1)

        # Update the weights
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features
            Arguments
            ---------
            features: 1D array of feature values
        '''
        # Hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


# mean squared error
def MSE(y, Y):  # Y is the true labels
    return np.mean((y - Y) ** 2)


def training(network, n_itrs, train_features, train_targets, val_features, val_targets):
    losses = {'train': [], 'validation': []}
    for i in range(n_itrs):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']  # only cnt as target

        network.train(X, y)

        # Printing out the training progress
        train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
        sys.stdout.write("\rProgress: {:2.1f}".format(100 * i / float(n_itrs))
                         + "% ... Training loss: " + str(train_loss)[:5]
                         + " ... Validation loss: " + str(val_loss)[:5])
        sys.stdout.flush()

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    _ = plt.ylim()

    return losses


# iterations = 10000
# learning_rate = 0.2
# hidden_nodes = 15
# output_nodes = 1
#
# N_i = train_features.shape[1]
# network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)


## plot
def predicting(network, scaled_features, test_features, test_targets, raw_data, test_data):
    fig, ax = plt.subplots(figsize=(8, 4))

    mean, std = scaled_features['cnt']
    predictions = network.run(test_features).T * std + mean
    ax.plot(predictions[0].clip(min=0), label='Prediction')
    ax.plot((test_targets['cnt'] * std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(raw_data.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)

