### Backpropagation in a Simple Neural Network

a)  
<img src="../1/dataset.png" title="" alt="dataset" width="453">

b)  
2. Derive the derivatives of Tanh, Sigmoid and ReLU  

$tanh(x)' = (\frac{e^x-e^{-x}}{e^x+e^{-x}})' = 1 - tanh^2(x)$ \  
$sigmoid(x)' = (\frac{1}{1+e^{-x}}) = \frac{e^{-x}}{(1+e^{-x})^2}$ \  
$relu(x)' = \begin{cases} 1\;\;\;& x \ge 0 \\ 0\;\;\; & x \lt 0 \end{cases}$

d)  

1. Derive the following gradients: $\frac{\partial L}{\partial W_1}$, $\frac{\partial L}{\partial b_1}$, $\frac{\partial L}{\partial W_2}$, $\frac{\partial L}{\partial b_2}$  

$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial a_1}\frac{\partial a_1}{\partial z_1}\frac{\partial z_1}{\partial W_1} = \frac{\partial L}{\partial z_2}W_2^T(actFun)'x^T$ \  
$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial a_1}\frac{\partial a_1}{\partial z_1}\frac{\partial z_1}{\partial b_1} = \frac{\partial L}{\partial z_2}W_2^T(actFun)'$ \  
$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial W_2} = \frac{\partial L}{\partial z_2}a_1^T$ \  
$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}\frac{\partial z_2}{\partial b_2} = \frac{\partial L}{\partial z_2}$

e)  

1. tanh  
   <img src="../1/tanh_2_3.png" title="" alt="tanh" width="508">  
   
   ```
   Loss after iteration 0: 0.211137  
   Loss after iteration 1000: 0.040499  
   Loss after iteration 2000: 0.042861  
   Loss after iteration 3000: 0.049291  
   Loss after iteration 4000: 0.049299  
   Loss after iteration 5000: 0.049310  
   Loss after iteration 6000: 0.049317  
   Loss after iteration 7000: 0.049322  
   Loss after iteration 8000: 0.049325  
   Loss after iteration 9000: 0.049327  
   Loss after iteration 10000: 0.049328  
   Loss after iteration 11000: 0.049328  
   Loss after iteration 12000: 0.049329  
   Loss after iteration 13000: 0.049329  
   Loss after iteration 14000: 0.049329  
   Loss after iteration 15000: 0.049329  
   Loss after iteration 16000: 0.049329  
   Loss after iteration 17000: 0.049329  
   Loss after iteration 18000: 0.049329  
   Loss after iteration 19000: 0.049329  
   ```

sigmoid
<img src="../1/sigmoid_2_3.png" title="" alt="sigmoid" width="510">  

```
Loss after iteration 0: 0.339078  
Loss after iteration 1000: 0.053047  
Loss after iteration 2000: 0.049460  
Loss after iteration 3000: 0.049229  
Loss after iteration 4000: 0.049204  
Loss after iteration 5000: 0.049199  
Loss after iteration 6000: 0.049193  
Loss after iteration 7000: 0.049185  
Loss after iteration 8000: 0.049178  
Loss after iteration 9000: 0.049173  
Loss after iteration 10000: 0.049168  
Loss after iteration 11000: 0.049164  
Loss after iteration 12000: 0.049160  
Loss after iteration 13000: 0.049158  
Loss after iteration 14000: 0.049156  
Loss after iteration 15000: 0.049154  
Loss after iteration 16000: 0.049153  
Loss after iteration 17000: 0.049152  
Loss after iteration 18000: 0.049152  
Loss after iteration 19000: 0.049151  
```

relu \  
<img src="../1/relu_2_3.png" title="" alt="relu" width="497">  

```
Loss after iteration 0: 0.094082  
Loss after iteration 1000: 0.036933  
Loss after iteration 2000: 0.036241  
Loss after iteration 3000: 0.036590  
Loss after iteration 4000: 0.037334  
Loss after iteration 5000: 0.036149  
Loss after iteration 6000: 0.035113  
Loss after iteration 7000: 0.036553  
Loss after iteration 8000: 0.035062  
Loss after iteration 9000: 0.036505  
Loss after iteration 10000: 0.036609  
Loss after iteration 11000: 0.036571  
Loss after iteration 12000: 0.036750  
Loss after iteration 13000: 0.036567  
Loss after iteration 14000: 0.036666  
Loss after iteration 15000: 0.036493  
Loss after iteration 16000: 0.036698  
Loss after iteration 17000: 0.036405  
Loss after iteration 18000: 0.036367  
Loss after iteration 19000: 0.035546  
```

2. hidden dim = 3  
   <img src="../1/tanh_2_3.png" title="" alt="" width="351">  

hidden dim = 5  
<img src="../1/tanh_2_5.png" title="" alt="" width="354">  

hidden dim = 10  
<img src="../1/tanh_2_10.png" title="" alt="" width="361">  

When hidden dim increases, the boundary fits the data better, and the loss decreases. 
However, when hidden dim is too large, the boundary becomes complex, which may cause overfitting.

f)  
1 hidden layer  
<img src="../1/deep_tanh_2_1.png" title="" alt="" width="365">  
2 hidden layers  
<img src="../1/deep_tanh_2_2.png" title="" alt="" width="367">  
3 hidden layers  
<img src="../1/deep_tanh_2_3.png" title="" alt="" width="369">  
4 hidden layers  
<img src="../1/deep_tanh_2_4.png" title="" alt="" width="369">  
5 hidden layers  
<img src="../1/deep_tanh_2_5.png" title="" alt="" width="370">  
6 hidden layers  
<img src="../1/deep_tanh_2_6.png" title="" alt="" width="370">  
7 hidden layers  
<img src="../1/deep_tanh_2_7.png" title="" alt="" width="375">  
8 hidden layers  
<img src="../1/deep_tanh_2_8.png" title="" alt="" width="378">  
9 hidden layers  
<img src="../1/deep_tanh_2_9.png" title="" alt="" width="379">  

When the number of hidden layers increases, the boundary fits the data better.
However, deep neural network may cause gradient vanishing, so 9 hidden layers performs worse.

### Training a Simple Deep Convolutional Network on MNIST

b) relu + adam

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/train.png" title="" alt="" width="565">

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/test.png" title="" alt="" width="569">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/conv1.biases.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/conv1.weights.png" alt="" width="286">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/conv2.biases.png" alt="" width="279"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/conv2.weights.png" alt="" width="280">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/conv1.activations_conv_relu.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/conv2.activations_conv_relu.png" alt="" width="283">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/fc1.activations_fc1_relu.png" alt="" width="285"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adam/maxpool2.activations_maxpool.png" alt="" width="285">

c) 

1. sigmoid + adam
   
   <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/train.png" title="" alt="" width="565">

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/test.png" title="" alt="" width="569">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/conv1.biases.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/conv1.weights.png" alt="" width="286">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/conv2.biases.png" alt="" width="279"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/conv2.weights.png" alt="" width="280">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/conv1.activations_conv_sigmoid.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/conv2.activations_conv_sigmoid.png" alt="" width="283">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/fc1.activations_fc1_sigmoid.png" alt="" width="285"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/sigmoid-adam/maxpool2.activations_maxpool.png" alt="" width="285">

2.
tanh + adam
<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/train.png" title="" alt="" width="565">

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/test.png" title="" alt="" width="569">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/conv1.biases.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/conv1.weights.png" alt="" width="286">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/conv2.biases.png" alt="" width="279"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/conv2.weights.png" alt="" width="280">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/conv1.activations_conv_tanh.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/conv2.activations_conv_tanh.png" alt="" width="283">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/fc1.activations_fc1_tanh.png" alt="" width="285"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/tanh-adam/maxpool2.activations_maxpool.png" alt="" width="285">

3. leakyrelu + adam
   
   <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/train.png" title="" alt="" width="565">

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/test.png" title="" alt="" width="569">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/conv1.biases.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/conv1.weights.png" alt="" width="286">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/conv2.biases.png" alt="" width="279"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/conv2.weights.png" alt="" width="280">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/conv1.activations_conv_leaky_relu.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/conv2.activations_conv_leaky_relu.png" alt="" width="283">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/fc1.activations_fc1_leaky_relu.png" alt="" width="285"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/leakyrelu-adam/maxpool2.activations_maxpool.png" alt="" width="285">

4. relu + sgd
   
   <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/train.png" title="" alt="" width="565">

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/test.png" title="" alt="" width="569">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/conv1.biases.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/conv1.weights.png" alt="" width="286">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/conv2.biases.png" alt="" width="279"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/conv2.weights.png" alt="" width="280">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/conv1.activations_conv_relu.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/conv2.activations_conv_relu.png" alt="" width="283">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/fc1.activations_fc1_relu.png" alt="" width="285"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-sgd/maxpool2.activations_maxpool.png" alt="" width="285">

5. relu + momentum
   
   <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/train.png" title="" alt="" width="565">

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/test.png" title="" alt="" width="569">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/conv1.biases.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/conv1.weights.png" alt="" width="286">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/conv2.biases.png" alt="" width="279"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/conv2.weights.png" alt="" width="280">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/conv1.activations_conv_relu.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/conv2.activations_conv_relu.png" alt="" width="283">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/fc1.activations_fc1_relu.png" alt="" width="285"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-momentum/maxpool2.activations_maxpool.png" alt="" width="285">

6. relu + adagrad
   
   <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/train.png" title="" alt="" width="565">

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/test.png" title="" alt="" width="569">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/conv1.biases.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/conv1.weights.png" alt="" width="286">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/conv2.biases.png" alt="" width="279"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/conv2.weights.png" alt="" width="280">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/conv1.activations_conv_relu.png" alt="" width="280"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/conv2.activations_conv_relu.png" alt="" width="283">

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/fc1.activations_fc1_relu.png" alt="" width="285"><img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW1/2/relu-adagrad/maxpool2.activations_maxpool.png" alt="" width="285">



different activation function + same optimizer: the weight and bias of each layer looks similar;

same activation function + different optimizer: the weight and bias of each layer differ.
