### Description of the model

1. For the regressor model, we use a MultiLayerNetwork and the PyTorch neural network library, as MultiLayerNetwork should give a better performance compared with single layer LinearRegression model and with functions provided by Pytorch neural network library, it is much easier and efficient to implement the training and evaluation steps.

2. Preprocessing Procedure:
   - missing values in the data, will be set to a default value: the mean of the data in that column.  
   - Preprocessor class in part1 is used to normalised the data, so the data is scaled to lie in the interval [0, 1].
   - Textual attribute in the data will be encoded to an array with only 0, 1 in it, using one-hot encoding. We replace the textual attribute in the dataset with the array
   - Data with type numpy.ndarray will be returned.



### Description of the evaluation setup.

<img src="https://c3.ai/wp-content/uploads/2020/11/Screen-Shot-2020-11-10-at-8.06.16-AM-500x159.png" alt="RMSE formula" style="zoom:50%;" />



### Information about the hyperparameter search

1. ***optimizer & loss function***

   <img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126162959134.png" alt="image-20211126162959134" style="zoom:50%;" />

   <img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126163030477.png" alt="image-20211126163030477" style="zoom:50%;" />

   Firstly, we want to figure out what will be the best combination of loss function and optimizer. For the optimizer list, we didn’t try too many optimiser algorithms in “torch.optim” and we just pick the most commonly used methods “Adam” and “SGD”. And for the loss function, we would like to choose between “MSE” and “MAE”. So we tried all the combinations and for every combination we train and test ten times. At the end we calculated the average regressor error to choose the one with the lowest error. Figure above shows our results of this part.

   The table shows that “Adam+MSE” performance best which leads to a lowest error. And we can find that using “Adam” results in a much better performance than using “SGD” clearly from the table. So we choose “Adam” and “MSE” as our optimizer and loss function respectively to continue to search other hyperparameter.

2. ***learning rate&epoch***

3. ***neuron number in each layer&activation function of each layer***

   

   

### Final evaluation of your best model

