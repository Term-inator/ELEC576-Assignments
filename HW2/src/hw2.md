1.

b.

batch size

| 64                                                                                                                                                                                     | 32                                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_19-52-12/1.png" alt="" width="468"> | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_19-55-13/1.png" title="" alt="" width="469"> |

It's hard to say which one is better, but bigger batch size trains faster, so here chooses 64



learning rate

| 0.001                                                                                                                                                                                  | 0.0005                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_19-52-12/1.png" alt="" width="468"> | <img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_20-08-34/1.png" alt="" width="470"> |

It's hard to say which one is better, but bigger learning rate trains faster, so here chooses 0.001



optimizer

| adam                                                                                                                                                                                   | sgd                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_19-52-12/1.png" alt="" width="348"> | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_20-13-54/1.png" title="" alt="" width="347"> |

It's obvious that adam converges faster, so here chooses adam



first fc layner nodes

| 120                                                                                                                                                                                    | 1024                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_19-52-12/1.png" alt="" width="712"> | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_20-19-03/1.png" title="" alt="" width="723"> |

It turns out that 120 has better accuracy on test set, so I changed the model.



c.

Final result:

batch_size: 64

epochs: 10

fc_1_n_nodes: 120

lr: 0.001

momentum: null

optimizer_type: adam



train/test accuracy and and train loss

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_20-24-09/1.png" title="" alt="" width="366">

first convolutional layer’s weights

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_20-24-09/conv1_weights.png" title="" alt="" width="366">

the statistics of the activations in the convolutional layers on test images

<img title="" src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/cifar10_Oct24_20-24-09/2.png" alt="" width="366">



2.

They present a novel way to map features back to the input pixel space, showing which part of an image contribute the most to features, revealing how CNN classifies images well. They use Deconvnet, Unpooling, Rectification to reverse the process of convolution, maxpooling and activation, and visualize features of different layers. This article shows what each layer have learned during training, and guide the training in return.



3.

a.

learning rate

| 0.01                                                                                                                                      | 0.001                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| ![](D:\University\Rice\2023Fall\COMP576%20Introduction%20to%20Deep%20Learning\HW\ELEC576-Assignments\HW2\runs\minst_Oct24_22-07-16\1.png) | ![](D:\University\Rice\2023Fall\COMP576%20Introduction%20to%20Deep%20Learning\HW\ELEC576-Assignments\HW2\runs\minst_Oct24_22-11-40\1.png) |

lr = 0.001 is much better



optimizer

| adam                                                                                                                                      | sgd                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| ![](D:\University\Rice\2023Fall\COMP576%20Introduction%20to%20Deep%20Learning\HW\ELEC576-Assignments\HW2\runs\minst_Oct24_22-07-16\1.png) | ![](D:\University\Rice\2023Fall\COMP576%20Introduction%20to%20Deep%20Learning\HW\ELEC576-Assignments\HW2\runs\minst_Oct24_22-14-46\1.png) |

sgd with momenton 0.9 is better



hidden size

| 64  | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-20-28/1.png" title="" alt="" width="380"> |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 128 | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-14-46/1.png" title="" alt="" width="379"> |
| 256 | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-18-04/1.png" title="" alt="" width="380"> |

We can see that when hidden size = 128, train loss converges faster and better.



Final result

epochs: 10

hidden_size: 128

lr: 0.001

optimizer_type: sgd

<img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-29-54/1.png" title="" alt="" width="441">



b.

lstm

| 64  | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-55-14/1.png" title="" alt="" width="361"> |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 128 | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-34-30/1.png" title="" alt="" width="362"> |
| 256 | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-53-08/1.png" title="" alt="" width="363"> |



gru

| 64  | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-47-09/1.png" title="" alt="" width="368"> |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 128 | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-37-28/1.png" title="" alt="" width="367"> |
| 256 | <img src="file:///D:/University/Rice/2023Fall/COMP576%20Introduction%20to%20Deep%20Learning/HW/ELEC576-Assignments/HW2/runs/minst_Oct24_22-49-29/1.png" title="" alt="" width="366"> |

1. hidden size larger, training loss converges faster

2. training loss of rnn recreases immediately, while lstm and gru's decrease slowly at first. 



c.

It's important for models to know what's around one pixel, because items on images are constructed by pixels. So, both CNN and RNN take pixels nearby into account. CNN use kernel to get features of an area, and RNN get features of a row. However, RNN's memory is bad, so for area consists 2 or more rows, it is hard for RNN to remember information got in the last row, which makes it hard for RNN to perform well in images. MINST is a simple dataset, so RNN can still have a good performance.
