### Description of the model

1. For the regressor model, we use a MultiLayerNetwork and the PyTorch neural network library, as MultiLayerNetwork should give a better performance compared with single layer LinearRegression model and with functions provided by Pytorch neural network library, it is much easier and efficient to implement the training and evaluation steps.

2. Preprocessing Procedure:
   - missing values in the data, will be set to a default value: the mean of the data in that column.  
   - Preprocessor class in part1 is used to normalised the data, so the data is scaled to lie in the interval [0, 1].
   - Textual attribute in the data will be encoded to an array with only 0, 1 in it, using one-hot encoding. We replace the textual attribute in the dataset with the array
   - Data with type numpy.ndarray will be returned.

### Description of the evaluation setup.

In this part we evaluated our neural network with “Root Mean Squared Error” (MSE), which calculates the score with the differences between values predicted and the actual desired values.

<img src="https://c3.ai/wp-content/uploads/2020/11/Screen-Shot-2020-11-10-at-8.06.16-AM-500x159.png" alt="RMSE formula" style="zoom:50%;" />

where y is the true value and ŷ is the predicted value. And N represents the total number of the data.

<img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126171546152.png" alt="image-20211126171546152" style="zoom:50%;" />

We chose RMSE as our score because it ranges from 0 to +infinity and indifferent to the direction of errors. It shows clearly the predication difference which is similar to the real house price, so that we can easily evaluated our model. We also tried MSE and MAE and recorded some of the results, which is showing in the table above. Compare with MAE, RMSE will be more sensitive to errors and gives a relatively high weight to large errors which will be more useful to evaluate our model since we don’t want large errors happening in our predictions.

We also divided the whole dataset into training dataset and validation dataset at a ratio of 8:2. So that we can train the model by the training dataset and use the validation dataset to evaluate the model.

### Information about the hyperparameter search

1. ***optimizer & loss function***

   <img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126162959134.png" alt="image-20211126162959134" style="zoom:50%;" />

   <img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126163030477.png" alt="image-20211126163030477" style="zoom:50%;" />

   Firstly, we want to figure out what will be the best combination of loss function and optimizer. For the optimizer list, we didn’t try too many optimiser algorithms in “torch.optim” and we just pick the most commonly used methods “Adam” and “SGD”. And for the loss function, we would like to choose between “MSE” and “MAE”. So we tried all the combinations and for every combination we train and test ten times. At the end we calculated the average regressor error to choose the one with the lowest error. Figure above shows our results of this part.

   The table shows that “Adam+MSE” performance best which leads to a lowest error. And we can find that using “Adam” results in a much better performance than using “SGD” clearly from the table. So we choose “Adam” and “MSE” as our optimizer and loss function respectively to continue to search other hyperparameter.

2. ***learning rate&epoch***

   <img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126180226297.png" alt="image-20211126180226297" style="zoom:50%;" />

   

   <img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126180645430.png" alt="image-20211126180645430" style="zoom:40%;" /><img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126180445705.png" alt="image-20211126180445705" style="zoom:40%;" /> 

   *Above is a zoom-in version*

   Now we use “Adam” and “MSE” as our optimizer and  loss function and trying to deal with the learning rate and the number of epoch hyper-parameters. For searching them, we tried different combinations between learning rate [0.0001, 0.001, 0.01, 0.02, 0.1, 0.2] and epoch_num [1000, 2000, … , 8000]. Every time we trained ten times and then get the average error. The results will be show in the graph above. We found out that larger number of epoch performance better but after reaching about 4000-5000, the performance stops improving and 3000 would be the best of our choice. We can also found that too small or too big value of learning rate will both increase the loss error, so we choose 0.01 which plays the most stable performance.

3. ***neuron number in each layer&number of layers***

   neurons_list = [[32,16,8,1], [48,24,12,6,1], [64,32,16,8,1], [64,32,16,8,4,1], [128, 64,32,16,8,1]]

   <!--neurons_list is the list of neurons where elem in it is the number of neurons in each linear layer represented as a list. The length of the list determines the number of linear layers -->

   Our strategy to arrange the neuron number in each layer is halve the neuron number with increasing of the layer number. 

   We tested the average error with different number of neurons in the first layer and with different layer number.

   The number of neurons in the first layer is guessed reasonably by trial and error.

    <img src="/Users/chen/Library/Application Support/typora-user-images/image-20211126183507192.png" alt="image-20211126183507192" style="zoom:50%;" />

   We use “Adam” and “MSE” as our optimizer and  loss function, ReLu is chosen as the activation function. 

   From the data we collected, we can see the model has the best performance with neurons:[32, 16, 8, 1]

4. ***activation function***

   Compared with Sigmoid function, there are two additional major benefits of ReLU fuction which make us choose ReLu as our final activation function:

   - *the reduced likelihood of the gradient to vanish:* This arises when 𝑎>0. In this regime the gradient has a constant value. In contrast, the gradient of sigmoids in this regime becomes increasingly small as the absolute value of x increases. The constant gradient of ReLUs results in faster learning.
   - *sparsity*：Sparsity arises when 𝑎≤0. The more such units that exist in a layer the more sparse the resulting representation. Sigmoid on the other hand are always likely to generate some non-zero value，resulting in dense representations. Sparse representations seem to be more beneficial than dense representations.

   

### Final evaluation of your best model

