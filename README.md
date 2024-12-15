<h1 align="center">Hi 👋, here we have the Neural Network labs</h1>
<h3 align="center">Developed these labs in the fifth semester of the faculty.</h3>


## Structure
* [Assignment_5: Developed and trained a Convolutional Neural Network for processing the Flappy Bird game using PyTorch framwork](#assignment-5)
* [Assignment_4.5: Developed and trained a Convolutional Neural Network (CNN) for processing the MNIST dataset using PyTorch framwork](#assignment-4-5)
* [Assignment_4: Developed and trained a Multi Layer Neural Network (MLNN) for processing the MNIST dataset using PyTorch framwork](#assignment-4)
* [Assignment_3: Developed and trained a Multi Layer Neural Network (MLNN) for processing the MNIST dataset using NumPy library](#assignment-3)
* [Assignment_2: First implementation of a single-layer Neural Network using NumPy library](#assignment-2)
* [Assignment_1: Familiarizing myself with the Numpy library](#assignment-1)


--------------------------------------------------------------------------------
<h1 id="assignment-5" align="left">Assignment_5:</h1>

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

<h3 align="left">The logic behind the code:</h3>
  - ...

<h3 align="left">How does it work?</h3>
  - ...

* [Structure](#structure)

---

<h1 id="assignment-4-5" align="left">Assignment_4.5:</h1>

<h3 align="left">Here we have the requirement:</h3>

Participate in the following Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870.
Use PyTorch to create a training Pipeline and do several experiments on the MNIST dataset, following the rules. 
IMPORTANT NOTE: Implement and train a CNN!

<h3 align="left">Specifications:</h3>

View the specifications on Kaggle competition -> https://www.kaggle.com/t/1422c0d3298e447aa6e50db3543b6870

<h3 align="left">The logic behind the code:</h3>

We will build the convolutional layers that will process the images from the MNIST dataset, which will be downloaded and uploaded in mini-batches, in order to then resize the outputs of these layers. Finally, we will build the fully connected layers that serve to provide the desired predictions. Finally we will apply the backpropagation process to adjust the weights/train the convolutional neural network

<h3 align="left">How does it work?</h3>

The convolutional neural network contains 2 convolutional layers followed by one Pooling layer each. The first convolution layer will contain 16 channels and the second layer will contain 32 channels. The transition from each convolution layer to the pooling layer is processed and filtered by means of a ReLU activation function and BatchNorm2d normalization.

Convolutional neural network achieves 99.50% test accuracy. The model that achieved this accuracy is saved in the file "best_model.pth". I also saved the prediction of the convolutional network in the file "submission.csv".

* [Structure](#structure)

---

<h1 id="assignment-4" align="left">Assignment_4:</h1>

<h3 id="assignment-4" align="left">Here we have the requirement:</h3>
(RO) ...

(EN) ...

<h3 align="left">The logic behind the code:</h3>
  - ...

<h3 align="left">How does it work?</h3>
  - ...

* [Structure](#structure)

---

<h1 id="assignment-3" align="left">Assignment_3:</h1>

<h3 align="left">Here we have the requirement:</h3>
(RO) ...

(EN) ...

<h3 align="left">The logic behind the code:</h3>
  - ...

<h3 align="left">How does it work?</h3>
  - ...

* [Structure](#structure)

---

<h1 id="assignment-2" align="left">Assignment_2:</h1>

<h3 align="left">Here we have the requirement:</h3>
(RO) ...

(EN) ...

<h3 align="left">The logic behind the code:</h3>
  - ...

<h3 align="left">How does it work?</h3>
  - ...

* [Structure](#structure)

---

<h1 id="assignment-1" align="left">Assignment_1:</h1>

<h3 align="left">Here we have the requirement:</h3>
(RO) ...

(EN) ...

<h3 align="left">The logic behind the code:</h3>
  - ...

<h3 align="left">How does it work?</h3>
  - ...

* [Structure](#structure)

---

<h3 align="left">Installation:</h3>

1. Clone the current repositoy! Now you have the project avalable on your local server!</br>
 Type command: ```git clone git@github.com:AndromedaOMA/Neural_Networks---Laboratories.git```
2. Select and open the chosen project through PyCharm IDE.
3. Have fun!
    
---

- ⚡ Fun fact **Through these labs I delved into the understanding and application of Neural Networks using both the NumPy (Numerical Python) library, for a detailed understanding of the concepts behind Neural Networks, and the PyTorch framework!**
