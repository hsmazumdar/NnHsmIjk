# NnHsmIjk
A Simple Neural Network with GUI

<img src=Readme_files/Ann.jpg>

Figure-1 Artificial Neural Network with Input, Hidden and Output neurons.


A multi-layer neural networks shown in figure-1  with backpropagation (BP) are simple and basic form of artificial neural network architecture. It consists of an input layer, a hidden layer, and an output layer.
  
<img src="Readme_files/AnnIJK.jpg">

Figure-2 A two-layer Neural Network with forward pass and back propagation training equations.

***************************************  
 
 


In figure-2 the input layer (Yi) receives the input data, which could be numerical or categorical features, and passes them forward to the hidden layer (Xj). The hidden layer performs a linear combination of the input values using weights (Wij) and applies a non-linear activation function to produce an output (Yj). Common activation functions include sigmoid, tanh, and ReLU. The purpose of the activation function is to introduce non-linearity into the network, allowing it to learn complex patterns and relationships in the data

The output layer (Xk) receives the transformed data from the hidden layer through wights and performs another linear combination using weights (Wjk). The final output of the network is obtained by applying an activation function to the result of the output layer (Yk). The activation function used in the output layer depends on the nature of the problem being solved. For example, if it is a binary classification problem, a sigmoid function might be used, while a softmax function could be used for multi-class classification.

  


During the training process, the network adjusts the weights using the backpropagation algorithm. Backpropagation calculates the gradients of the loss function with respect to the weights of the network. It then updates the weights by taking small steps (eta) in the opposite direction of the gradients, using an optimization algorithm such as stochastic gradient descent (SGD).

The number of neurons in the hidden layer is a hyper parameter that needs to be determined based on the complexity of the problem and the amount of available data. Too few neurons may result in under fitting, while too many neurons can lead to overfitting. Finding an appropriate number of neurons often requires experimentation and tuning.


<img src="Readme_files/AnnBP.jpg">
Figure-3 Neural Network Backpropagation Algorithm equations

***************************************  




Overall, a two-layer neural network with backpropagation shown in figure-3 is a basic and widely used architecture for solving various machine learning problems. While it is relatively simple compared to more complex neural network architectures, it can still be effective for tasks with moderate complexity.

Quick Start Steps-

1. Download the zip file and unzip in a folder NnHsmIjk.
2. Select NnLib.py and NnXor.py files and load in VS Code
3. Install necessary library components
4. Run NnXor.py to popup NnHsmIjk application of figure-4
5. Press 'Set' button for default NN configuration In:10-Hdn:7-Out:5
6. Press 'Save Net' button to open file save dialoug box of Figure-5
7. Enter the save net file name in dialoug box of Figure-5 and press 'Save' button
8. Press 'Load Net' button of figure-6 to load the saved net file
9. Press 'Save Data' button to open file save dialoug box of Figure-7
10. Enter the save data file name in dialoug box of Figure-7 and press 'Save' button,
    this will save default 1000 data records in saved file
11. Press 'Load Data' button of main application of figure-4 to load the saved data file
12. Press 'Reset' to reset all weights to random values
13. Press 'Train' button to train the Neural Net. 
    Error plot will display the training status shown in figure-9 


 
<img src="Readme_files/Main.png">
Figure-4 A Simple Neural Network Application

***************************************  
  




<img src="Readme_files/SaveNet.png">
Figure-5 Save Neural Network File Dialog Box

***************************************  




<img src="Readme_files/LoadNet.png">

Figure-6 Load Neural Network File Dialog Box

***************************************  




  
<img src="Readme_files/SaveData.png">

Figure-7 Save Data File Dialog Box

***************************************  


<img src="Readme_files/LoadData.png">

Figure-8 Load Data File Dialog Box

***************************************  

<img src="Readme_files/Main2.png">

Figure-9 Trained Neural Network Application 

***************************************  
 
 


  


