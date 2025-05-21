<h1 align="center">Hi 👋, here we have the Neural Network labs</h1>
<h3 align="center">Deep NN Architectures: Deep Q NN (DQN) - Residual NN (ResNet) - Convolutional NN (CNN) - Linear/Multilateral NN (MLNN) </h3>


## Table Of Content
* [Assignment_5: Developed and trained a Deep Q-Learning using Convolutional Neural Network for processing the Flappy Bird game using PyTorch framework](#assignment-5)
* [Assignment_4.5.2: Developed and trained a Residual Neural Network (ResNet) for processing the MNIST dataset using PyTorch framwork](#assignment-4-5-2)
* [Assignment_4.5: Developed and trained a Convolutional Neural Network (CNN) for processing the MNIST dataset using PyTorch framwork](#assignment-4-5)
* [Assignment_4: Developed and trained a Multi Layer Neural Network (MLNN) for processing the MNIST dataset using PyTorch framwork](#assignment-4)
* [Assignment_3: Developed and trained a Multi Layer Neural Network (MLNN) for processing the MNIST dataset using NumPy library](#assignment-3)
* [Assignment_2: First implementation of a single-layer Neural Network (Peceptron) using NumPy library](#assignment-2)
* [Assignment_1: Familiarizing myself with the Python programming language](#assignment-1)
* [Installation](#inst)

--------------------------------------------------------------------------------
<h1 id="assignment-5" align="left">Assignment_5: FlappyNet</h1>

<h3 align="left">Here we have the requirement:</h3>

Implement and train a neural network using the Q learning algorithm to control an agent in the Flappy Bird game.

<h3 align="left">Environment:</h3>

You can use a Flappy Bird environment from [here](https://pypi.org/project/flappy-bird-gymnasium/) or [here](https://github.com/Talendar/flappy-bird-gym) or another environment. Why not create your own? If you use a pre-made environment, make sure you can render the environment and interact with it.

<h3 align="left">Specifications:</h3>

You can train the model directly on images (the model receives the pixels) or you can extract helpful features. Based on the input you are using for the model, the maximum score is capped to:

- 20 points: if you provide the game state directly (this might include positions of the pipes, bird, direction, simple distances)
- 25 points: if you provide preprocessed features (this might include more complex features extracted from the image: e.g. sensors/lidar for the bird)
- 30 points: if you use the image as input, eventually preprocessed, if needed (resizing, grayscale conversion, thresholding, dilation, erosion, background removal, etc.)

It is not necessary to implement the neural network from scratch (you can use PyTorch), but you must implement the Q learning algorithm.

<h3 align="left">How does it work?</h3>

  Implemented several scripts that serve the purpose of training the FlappyBird agent. The scripts are placed in two broad typologies: Deep Q Learning (DQL) and Convolutional Deep Q Learning (CDQN). Both follow the same implementation, the only difference is given by the convolutional aspect of CDQN.
  
  At the same time, for each type of architecture, pre-trained models are available from which we can choose to visualize the FlappyBird agent through the graphical interface, through the "FlappyBird-v0" environment for Gymnasium.
  
  Delving into the implementation idea, the solution uses the concepts of Double DQN (DDQN) and Dueling DQN (Dueling Architecture). These approaches significantly help in training the agent.

  
<h3 align="left">The logic behind the code:</h3>

  The final solution presents the implementation of Neural Networks that are focused on Reinforcement Learning concepts through the Q-Learning algorithm. The first implementation ideas will consider only some classic Neural Networks (without Dueling architecture). So these were the initial implementations of Fully Connected, respectively Convolutional Neural Networks.

  The multilayer neural network consists of three hidden layers of 256 perceptrons each followed by a LayerNorm for data normalization. For the first layer we will use the ReLU activation function, and for the second layer the GeLU activation function and a Dropout for regularization.
  
  The convolutional neural network contains 2 convolutional layers followed by one Pooling layer each. The first convolution layer will contain 16 channels and the second layer will contain 32 channels. The transition from each convolution layer to the pooling layer is processed and filtered by means of a ReLU activation function and BatchNorm2d normalization. The convolutional layers are followed by a fully connected network consisting of two hidden layers, between which is a ReLU activation function and the Dropout regularizer. The LogSoftmax function is attached to the last layer.

  Through these two Neural Networks, for each one separately, a Target Neural Network identical to the initial one was applied and trained, in parallel, through the Experience Replay stack.

  Finally the original Neural Networks were adapted to the Dueling Architecture. Dueling DQN and Dueling CNN Neural Networks receive the same specific architecture, the only significant difference between them is given by the convolutional layers described above. We use Dueling architecture because this architecture leads to better policy evaluation. NOTE: The Dueling architecture is nothing but an extension of the standard Deep Q-Network (DQN)!
  
  The architecture used will change the last layers of Fully Connected Neural Networks into two streams, one associated with value (value_stream) and another associated with advantages (advantages_stream). streams will consist of two hidden layers and a ReLU activation function.
  
  Here we have the summary and the function/formula we will apply within the Dueling architecture: 
```bash
Q(s,a) = V(s) + A(s,a) − (1/|A|) * ∑ A(s,a')
```
 I other words: V + A - mean(A), where V is the computed Value using the value_stream and A is the computed Advantages using the advantages_stream.

---

<h3 id="inst" align="left">Installation:</h3>

1. Clone the current repositoy! Now you have the project avalable on your local server!</br>
 Type command: ```git clone git@github.com:AndromedaOMA/FlappyNet.git```
2. Select, open and run the FlappyNet project through PyCharm IDE or the preferred IDE.
3. Have fun!

---

<h3 id="score" align="left">Best score:</h3>

<img src="https://github.com/user-attachments/assets/1b53b7f2-bc87-4ee5-ae97-14bdf6a11f06" alt="Moments before the disaster" style="width: 300px; height: auto;">

---

**NOTE**: This project represents the final project supported and realized within the Neueonal Networks laboratories of the Faculty of Computer Science of the Alexandru Ioan Cuza University in Iași, Romania.

**ALSO NOTE**: Developed this project together with Marin Andrei (andier13 on GitHub) in the fifth semester of the faculty.

---

- ⚡ Fun fact: **Through this project I developed better the subtle concepts of Reinforcement Learning and Q-Learning!**

* [Table Of Content](#table-of-content)

---

<h1 id="assignment-4-5-2" align="left">Assignment_4.5.2:</h1>

<h3 align="left">Here we have the requirement:</h3>

Participate in the following Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870.
Use PyTorch to create a training Pipeline and do several experiments on the MNIST dataset, following the rules. 

IMPORTANT NOTE: Implement and train a Residual Neural Network(ResNet)!
</br>
ALSO NOTE: This assignment is actually a bonus and continuation of assignment_4!

<h3 align="left">Specifications:</h3>

View the specifications on Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870

<h3 align="left">How does it work?</h3>

Built and implemented the residual blocks, which in turn use the BatchNorm2d normalization and the GELU activation function, that will process the images from the MNIST dataset, which will be downloaded and uploaded in mini-batches, in order to then resize the outputs of these layers. Also built the fully connected layers that serve to provide the desired predictions. Finally the backpropagation process will be applied to adjust the weights/train of the ResNet.

<h3 align="left">The logic behind the code:</h3>

The ResNet contains one convolutional layer of 16 channels followed by a 2D batch normalization and ReLU activation function, further, two residual blocks are implemented, each with 16 or 32 channels, the last one having a stride equal to 2. To make this possible, a second script "residual_block.py" will help implement the residual blocks that will take into account the residual connections that "skip" one or more layers.

The fully connected neural network contains a number of two hidden layers that help to process the prediction of the entire artificial neural network. Between the two layers we used a Dropout regularization to reduce the possibility of overfitting and the ReLU activation function.

From the optimization point of view, we will use the Adam optimizer and the CosineAnnealingWarmRestarts learning rate scheduler, which adjusts the learning rate using a cosine annealing strategy with periodic restarts. This technique helps escape local minima by periodically increasing the learning rate back to a higher value.

The data set was processed/transformed using RandomAffine for easy modification of these data, converted to tensors and normalized.

The Residual Neural Network achieves 99.71% test accuracy. The model that achieved this accuracy is saved in the file "best_model.pth". I also saved the prediction of the convolutional network in the file "submission.csv".

* [Table Of Content](#table-of-content)

---

<h1 id="assignment-4-5" align="left">Assignment_4.5:</h1>

<h3 align="left">Here we have the requirement:</h3>

Participate in the following Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870.
Use PyTorch to create a training Pipeline and do several experiments on the MNIST dataset, following the rules. 

IMPORTANT NOTE: Implement and train a Convolutional Neural Network(CNN)!
</br>
ALSO NOTE: This assignment is actually a bonus and continuation of assignment_4!

<h3 align="left">Specifications:</h3>

View the specifications on Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870

<h3 align="left">How does it work?</h3>

Built and implemented the convolutional layers that will process the images from the MNIST dataset, which will be downloaded and uploaded in mini-batches, in order to then resize the outputs of these layers. Also built the fully connected layers that serve to provide the desired predictions. Finally the backpropagation process will be applied to adjust the weights/train of the CNN.

<h3 align="left">The logic behind the code:</h3>

The convolutional neural network contains 2 convolutional layers followed by one Pooling layer each. The first convolution layer will contain 16 channels and the second layer will contain 32 channels. The transition from each convolution layer to the pooling layer is processed and filtered by means of a ReLU activation function and BatchNorm2d normalization.

The fully connected neural network contains a number of two hidden layers that help to process the prediction of the entire artificial neural network. Between the two layers we used a Dropout regularization to reduce the possibility of overfitting and the ReLU activation function.

From the optimization point of view, we will use the Adam optimizer and the CosineAnnealingWarmRestarts learning rate scheduler, which adjusts the learning rate using a cosine annealing strategy with periodic restarts. This technique helps escape local minima by periodically increasing the learning rate back to a higher value.

The data set was processed/transformed using RandomAffine for easy modification of these data, converted to tensors and normalized.

The Convolutional Neural Network achieves 99.56% test accuracy. The model that achieved this accuracy is saved in the file "best_model.pth". I also saved the prediction of the convolutional network in the file "submission.csv".

* [Table Of Content](#table-of-content)

---

<h1 id="assignment-4" align="left">Assignment_4:</h1>

<h3 align="left">Here we have the requirement:</h3>

Participate in the following Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870.
Use PyTorch to create a training Pipeline and do several experiments on the MNIST dataset, following the rules. 
IMPORTANT NOTE: Implement and train a Multy Layer Neural Network(MLNN)!

<h3 align="left">Specifications:</h3>

View the specifications on Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870

<h3 align="left">How does it work?</h3>

We will download and upload the MNIST dataset data in mini-batches to a classical fully connected neural network. Finally we will apply the backpropagation process to adjust the weights/train the neural network.

<h3 align="left">The logic behind the code:</h3>

The neural network contains a number of five hidden layers that help to process the prediction of the entire artificial neural network. Between these layers we will use Dropout regularizations to reduce the possibility of overfitting and LeakyReLU activation functions, as well as BatchNorm1d normalization to speed up the training process.

From the optimization point of view, we will use the Adam optimizer and the OneCycleLR learning rate scheduler, which adjusts the learning rate and optionally momentum during training to achieve faster convergence and better generalization.

The data set was processed/transformed using RandomAffine for easy modification of these data, converted to tensors and normalized.

The neural network achieves 99.40% test accuracy. The model that achieved this accuracy is saved in the file "best_model.pth". I also saved the prediction of the convolutional network in the file "submission_99.40.csv".

* [Table Of Content](#table-of-content)

---

<h1 id="assignment-3" align="left">Assignment_3:</h1>

<h3 align="left">Here we have the requirement:</h3>

Implement a Multi Layer Perceptron (MLP) and the Backpropagation algorithm
using NumPy operations. 

The MLP architecture should consist of 784 input
neurons, 100 hidden neurons, and 10 output neurons. 

Use the MNIST dataset
to evaluate your implementation. Measure the accuracy for both training and
validation. The tasks for this assignment and their respective points are as
follows:
  1. Implement forward propagation. (1 points)
  2. Implement the backpropagation algorithm using the chain rule. Refer to
  the provided resources. (15 points implementation + explanation)
  3. Utilize batched operations for enhanced efficiency. (4 points)
  4. You must achieve at least 95% accuracy on the validation split of the
  MNIST dataset, in 5-6 minutes. (5 points)
  5. Choose one: (5 points)
     
    (a) Implement dropout on the hidden layer.
    
    (b) Implement L1 or L2 regularization.
    
    (c) Create a dynamic learning rate scheduler that decays the learning rate when training metrics reach a plateau.

Things you may try to improve the accuracy:

  • Weight initialization
  
  • Changing the learning rate
  
  • Normalizing and shuffling the training data
  
  • Dropout and other forms of regularization
  
Bonus : Strive for higher accuracy (not graded, but always try to be better).

<h3 align="left">How does it work?</h3>

We have implemented and trained a first Neural Network using the NumPy library. It comes with a simplistic and quite efficient content, as the statement above requires: an input layer (784), a hidden layer (100) and an output layer (10).

Every neural layer, method, optimizer and activation function will be implemented from scratch.

<h3 align="left">The logic behind the code:</h3>

The multi layer network uses a ReLU activation function for modeling and processing and Softmax which will adjust the final results during the Forward propagation process. Time during which the partial derivatives of these functions will be used during the Back propagation process.

The weights will be initialized using the He method to avoid blocking the neural network. It helps stabilize the variance of activations across layers, leading to better convergence during training.

From the dataset point of view, I have implemented a method that will generate batches from the entire dataset and shuffle them.

The Neural Network achieves 97% test accuracy.

* [Table Of Content](#table-of-content)

---

<h1 id="assignment-2" align="left">Assignment_2:</h1>

<h3 align="left">Here we have the requirement:</h3>

In this exercise, you are tasked with implementing both the forward and backward
propagation processes for a neural network with 784 inputs and 10 outputs
using NumPy. This network can be thought of as consisting of 10 perceptrons,
each responsible for predicting one of the 10 output classes.

Given an input matrix X of shape (m, 784), where m is the batch size and 784
is the number of features (input neurons), a weight matrix W of shape (784, 10),
and a bias matrix b of shape (10, ), compute the output of the network for each
example in the batch, calculate the error, and update the weights and biases
accordingly.

Download the MNIST dataset, load the images, and propagate them through
your network. Record the initial prediction accuracy before training and the
prediction accuracy after training for a specified number of epochs.

1. Load the MNIST dataset.
2. Normalize the data and convert the labels to one-hot-encoding.
3. Train the perceptron for 50-500 epochs.
      • For each epoch, split the training data and training labels into batches of 100 elements.
      • For each batch, compute the output of the network using the softmax function. (3 points)
      • Implement the gradient descent algorithm to update the weights and biases. (7 points)
      • Have an efficient implementation. (2 points)
      • Achieve at least 90% accuracy on the testing data. (3 points)

<h3 align="left">The logic behind the code:</h3>

  We have here an introductory implementation of a perceptron, where we dealt with the simplest neural network typology.

* [Table Of Content](#table-of-content)

---

<h1 id="assignment-1" align="left">Assignment_1:</h1>

<h3 align="left">Here we have the requirement:</h3>

We have nothing more than simple and introductory problems in the world of Python.

* [Table Of Content](#table-of-content)

---

<h3 id="inst" align="left">Installation:</h3>

1. Clone the current repositoy! Now you have the project avalable on your local server!</br>
 Type command: ```git clone git@github.com:AndromedaOMA/Neural_Networks---Laboratories.git```
2. Select, open and run the chosen project through PyCharm IDE or the preferred IDE.
3. Have fun!
    
---

- ⚡ Fun fact **Through these labs I delved into the understanding and application of Neural Networks using both the NumPy (Numerical Python) library, for a detailed understanding of the concepts behind Neural Networks, and the PyTorch framework!**
