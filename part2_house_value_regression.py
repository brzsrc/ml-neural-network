from part1_nn_lib import Preprocessor
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.options.mode.chained_assignment = None  # default='warn'


class Regressor():

    def __init__(self, x, nb_epoch=1000, model=None, criterion=None, optimiser=None):
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

        # Replace this code with your own
        self.x_preprocessor = None
        # self.y_preprocessor = None
        self.label_dict = None
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        # the trained model
        self.model = model
        # to update parameters[set lr]
        self.optimiser = optimiser
        # the loss function
        self.criterion = criterion
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

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        # return x, (y if isinstance(y, pd.DataFrame) else None)
        col_attributes = x.dtypes[x.dtypes != 'object'].index

        # fill NA with the mean value of the column
        x[col_attributes] = x[col_attributes].fillna(x[col_attributes].mean())
        x = x.values
        # print(x)
        col = x[:, -1]
        col_left = x[:, :-1]
        # col_unique = list(np.unique(x[:,-1]))
        # lb = LabelBinarizer()
        # col_bin = lb.fit_transform(col_unique)
        # col_dict = dict(zip(col_unique, col_bin))
        # col_trans = [col_dict[elem] for elem in col]
        # col_final = np.concatenate((col_left, col_trans), axis=1)

        # preprocessor = Preprocessor(col_final)

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
            # print("y:", y)
            #self.y_preprocessor = Preprocessor(y)
            #data_y = self.y_preprocessor.apply(y)
            # print("data_y: ", data_y)
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
        # print(X.dtype)
        # print("Y = ", Y)
        # X = X.astype(float)
        # Y = Y.astype(float)
        x_train_tensor = torch.from_numpy(X).float()
        y_train_tensor = torch.from_numpy(Y).float()
        #build network
        self.model = LinearRegression(np.shape(X)[1], n_output_vars=1)
        #set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        #set loss function
        self.criterion = nn.MSELoss()
        for i in range(self.nb_epoch):
            #train
            self.model.train()
            y_pred = self.model(x_train_tensor)
            # Reset the gradients
            self.optimizer.zero_grad()
            # compute loss
            loss = self.criterion(y_pred, y_train_tensor)
            # Backward pass (compute the gradients)
            loss.backward()
            # update parameters
            self.optimizer.step()

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
        # print(x_predicted_tensor)
        y_predicted_tensor = self.model(x_predicted_tensor)
        # print(y_predicted_tensor)
        y_predicted = y_predicted_tensor.detach().numpy()
        # print(y_predicted)
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
        y_validation_tensor = torch.from_numpy(Y).float()
        # print(x_predicted_tensor)
        y_predicted = self.model.forward(x_predicted_tensor)
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


def RegressorHyperParameterSearch():
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

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def LinearRegression(n_input_vars, n_output_vars=1):
        model = nn.Sequential(
            nn.Linear(n_input_vars, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1))

        return model




def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=0)

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=7500)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
