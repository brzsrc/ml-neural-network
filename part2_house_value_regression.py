from part1_nn_lib import Preprocessor
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'


class Regressor():

    def __init__(self, x, nb_epoch=1000, neurons=[32,16,8,1], activations=["relu", "relu", "relu", "identity"], 
        criterion=nn.MSELoss(), optimiser_name=None, learning_rate=None):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

            input_size == # of features

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.x_preprocessor = None
        self.label_dict = None
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.neurons = neurons
        self.activations = activations
        # the trained model
        self.model = LinearRegression(np.shape(X)[1], neurons=self.neurons, activations=self.activations)
        # to update parameters[set lr]
        self.optimiser_name = optimiser_name
        # the loss function
        self.criterion = criterion
        self.learning_rate = learning_rate
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Return preprocessed x and y, return None for y if it was None
        col_attributes = x.dtypes[x.dtypes != 'object'].index

        # fill NA with the mean value of the column
        x[col_attributes] = x[col_attributes].fillna(x[col_attributes].mean())
        x = x.values
        col = x[:, -1]
        col_left = x[:, :-1]

        if training:
            col_unique = list(np.unique(x[:, -1]))
            lb = LabelBinarizer()
            col_bin = lb.fit_transform(col_unique)
            self.label_dict = dict(zip(col_unique, col_bin))
            col_trans = [self.label_dict[elem] for elem in col]
            col_final = np.concatenate((col_left, col_trans), axis=1)
            self.x_preprocessor = Preprocessor(col_final)
        else:
            col_trans = [self.label_dict[elem] for elem in col]
            col_final = np.concatenate((col_left, col_trans), axis=1)

        data_x = self.x_preprocessor.apply(col_final)
        data_x = data_x.astype(float)
        data_y = None

        # fill NA with the mean value of the column
        if y is not None:
            y = y.fillna(y.mean())
            y = y.values
            data_y = y.astype(float)

        return data_x, data_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        x_train_tensor = torch.from_numpy(X).float()
        y_train_tensor = torch.from_numpy(Y).float()

        lr = self.learning_rate if self.learning_rate is not None else 0.01

        if self.optimiser_name == 'Adam':
            optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif self.optimiser_name == 'SGD':
            optimiser = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            #Default optimiser
            optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        for i in range(self.nb_epoch):
            y_pred = self.model(x_train_tensor)
            # Reset the gradients
            optimiser.zero_grad()
            # compute loss
            loss = self.criterion(y_pred, y_train_tensor)
            # Backward pass (compute the gradients)
            loss.backward()
            # update parameters
            optimiser.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        x_predicted_tensor = torch.from_numpy(X).float()
        y_predicted_tensor = self.model(x_predicted_tensor)
        y_predicted = y_predicted_tensor.detach().numpy()
        return y_predicted

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        x_predicted_tensor = torch.from_numpy(X).float()
        y_predicted = self.model(x_predicted_tensor)
        y_predicted = y_predicted.detach().numpy()
        loss = mean_squared_error(Y, y_predicted)**0.5
        return loss  # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(regressor, x_train, x_test, y_train, y_test, 
    criterion_list, optimiser_list, learning_rate_list,
    nb_epoch_list, neurons_list, activations_list):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    for criterion in criterion_list:
        for optimiser in optimiser_list:
            i = 0
            totalErrors = 0.0
            while i < 5:
                regressor.criterion = criterion
                regressor.optimiser_name = optimiser
                regressor.fit(x_train, y_train)
                error = regressor.score(x_test, y_test)
                print("i :", i)
                print("\nRegressor error: {}\n".format(error))
                totalErrors += error
                i += 1

            avgError = totalErrors / 5
            print("avgError :", avgError)

    for nb_epoch in nb_epoch_list:
        i = 0
        totalErrors = 0.0
        regressor.nb_epoch = nb_epoch
        while i < 10:
            regressor.fit(x_train, y_train)
            error = regressor.score(x_test, y_test)
            print("i :", i)
            print("\nRegressor error: {}\n".format(error))
            totalErrors += error
            i += 1

        avgError = totalErrors / 10
        print("avgError :", avgError)

    for lr in learning_rate_list:
        regressor.learning_rate = lr
        print("learning_rate = ", lr)
        for nb_epoch in nb_epoch_list:
            i = 0
            totalErrors = 0.0
            regressor.nb_epoch = nb_epoch
            while i < 10:
                regressor.fit(x_train, y_train)
                error = regressor.score(x_test, y_test)
                # print("i :", i)
                # print("\nRegressor error: {}\n".format(error))
                totalErrors += error
                i += 1
            avgError = totalErrors / 10
            print("nb_epoch = ", nb_epoch)
            print("avgError :", avgError)

    for i in range(len(neurons_list)):
        regressor.neurons = neurons_list[i]
        regressor.activations = activations_list[i]
        k = 0
        totalErrors = 0.0
        while k < 10:
            regressor.fit(x_train, y_train)
            error = regressor.score(x_test, y_test)
            # print("k :", k)
            # print("\nRegressor error: {}\n".format(error))
            totalErrors += error
            k += 1
        avgError = totalErrors / 10
        print("avgError :", avgError)

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

class LinearRegression(nn.Module):
    def __init__(self, n_input_vars, neurons=None, activations=None):
        super().__init__()  # call constructor of superclass
        """
        - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
        - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.neurons = neurons
        self.activations = activations

        self.layers = nn.ModuleList()
        # The first layer should between the input and the first neuron
        self.layers.append(nn.Linear(n_input_vars, self.neurons[0]))
        for i in range(len(neurons) - 1):
            layer = nn.Linear(neurons[i], neurons[i + 1])
            self.layers.append(layer)

    def forward(self, x):
        output = x
        for i in range(len(self.layers)):
            # Add activation layer if exist
            if self.activations[i] == "sigmoid":
                output = torch.sigmoid(self.layers[i](output))
            elif self.activations[i] == "relu":
                output = torch.relu(self.layers[i](output))
            else:
                output = self.layers[i](output)
        return output




def example_main():
    
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Spliting dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=0)

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting

    # neurons_list=[[32,16,8,1], [64,32,16,8,1], [48,24,12,6,1]]
    # activations_list=[["relu", "relu", "relu", "identity"], ["relu", "relu", "relu", "identity"], ["relu", "relu", "relu", "identity"]]
    neurons_list=[[64,32,16,8,1], [128, 64, 32, ], [], []]
    activations_list=[["relu", "relu", "relu", "relu", "relu"]]
 
    nb_epoch_list=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    criterion_list = [nn.MSELoss(), nn.L1Loss()]
    optimiser_list = ["Adam", "SGD"]
    learning_rate_list = [0.0001, 0.001, 0.01, 0.02, 0.1, 0.2]
    regressor = Regressor(x_train)
    # RegressorHyperParameterSearch(regressor, x_train, x_test, y_train, y_test, 
    #     nb_epoch_list=nb_epoch_list, criterion_list=criterion_list, 
    #     optimiser_list=optimiser_list, learning_rate_list=learning_rate_list,
    #     neurons_list=neurons_list, activations_list=activations_list)
    
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()



