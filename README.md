Sample images from IQ-OTH/NCCD, LC25000 and LIDC-IDRI datasets, diagnosed as either lung cancer or healthy.
![image](https://github.com/user-attachments/assets/ba384f60-3329-4eb5-9fb0-4901aac36e97)

An overview of the proposed system.
![image](https://github.com/user-attachments/assets/585a5da0-dc18-40d2-a78c-6f5f8747b0ec)



	


3.1 Datasets used
In this research, three popular datasets which are publicly available, are considered for training and evaluating the proposed system. These datasets are relatively small and provide the necessary challenges that tend to pose limitations of the DL models. The details of the considered datasets are provided in the following sub-sections:
3.1.1 IQ-OTH/NCCD
It comprises CT scans of patients at various stages of lung cancer and healthy subjects, marked by oncologists and radiologists. The dataset consists of 1190 CT scan images from 110 cases, categorized into three classes: normal (55 cases), benign (15 cases), and malignant (40 cases). Table 1 shows the organization of the IQ-OTH/NCCD dataset.
Table 1. Data composition of the IQ-OTH/NCCD dataset.
Class	Training Set	Validation Set	Test Set
Benign	97	11	12
Malignant	450	47	64
Normal	330	52	34
3.1.2 LC25000
This dataset consists of 25,000 histopathological images, divided into five distinct classes. We consider only three classes namely, Lung Benign tissue, Lung Adenocarcinoma and Lung Squamous cell carcinoma. As depicted in Table 2, each class contains ≈ 5000 images divided into 80% training set while validation and test sets having 10% images each. All images are 768×768 pixels in size and are in jpeg file format. The dataset contains histopathological images of lung and colon tissues. Since our work is centered around lung cancer, we would only be considering the lung image set comprising 15000 images.
Table 2. Data composition of LC25000 for images under lung cancer classes only.
Class	Training Set	Validation Set	Test Set
Lung Adenocarcinoma	3893	513	504
Lung Benign	4014	492	494
Lung Squamous cell carcinoma	4003	495	502
3.1.3 LIDC-IDRI
This dataset consists of diagnostic and lung cancer screening thoracic CT scans with marked-up annotated lesions. It is a web-accessible international resource for development, training, and evaluation of CAD methods for lung cancer detection and diagnosis.
Table 3. Data composition of the LIDC-IDRI dataset.
Class	Training Set	Validation Set	Test Set
Malignant	673	168	210
Benign	650	162	203
As shown in Table 3, the dataset consists of 2066 images distributed across a train set of 1323 images, a test set of 413 images and a validation set of 330 images. Each set further has data distributed across 3 subclasses viz., benign and malignant. This dataset comprises diagnostic and lung cancer screening thoracic CT scans with annotated lesions


3.2 Pre-trained DL Models
Here, two base models called MobileNetV2 and DenseNet121 are explored for understanding the efficiency in lung cancer identification. In both the cases, an input image of size 224×224×3 from the lung cancer image dataset 

mobilenet
In MobileNetV2, there are two types of blocks. As shown in Fig. 3, one is residual block with stride of and Another one is block with stride of 2 for downsizing. There are 3 layers for both types of blocks. This time, the first layer is 1×1 convolution with ReLU6. The second layer is the depthwise convolution. The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if ReLU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain. And there is an expansion factor t. And t=6 for all main experiments. If the input got 64 channels, the internal output would get 64×t=64×6=384 channels where t: expansion factor, c: number of output channels, n: repeating number, s: stride. 3×3 kernels are used for spatial convolution. In typical use, the primary network (width multiplier 1, 224×224), has a computational cost of 300 million multiply-adds and uses 3.4 million parameters. The performance tradeoffs are further explored, for input resolutions from 96 to 224, and width multipliers of 0.35 to 1.4. The network computational cost up to 585M MAdds, while the model size varies between 1.7M and 6.9M parameters. To train the network, 16 GPUs are used with a batch size of 96.
![image](https://github.com/user-attachments/assets/3bedb966-2c5d-4c2f-b4bc-5ffc5239e5bc)


dense
In DenseNet121, there are two types of layers: convolutional layers within dense blocks and transition layers for down-sampling as depicted in Fig. 4. Each dense block consists of multiple layers where each layer receives input from all preceding layers. There are 3 key operations in each dense block layer. The first is a 1×1 convolution followed by ReLU. The second is a 3×3 convolution, and the third operation is concatenation of the input with its preceding layers’ outputs, promoting feature reuse. There is a growth rate k that determines the number of output channels per layer. For all main experiments, k is typically set to 32. If the input has 64 channels, after one layer, the output would have 64 + k = 64 + 32 = 96 channels. Transition layers, which include a 1×1 convolution followed by 2×2 average pooling, are used for down-sampling between dense blocks. These layers help control the complexity of the network and manage the size of the feature maps.
![image](https://github.com/user-attachments/assets/4c89e2a9-b02e-460b-ab8b-8f16e000e521)





3.3 Soft Attention Scheme
As the primary aim of our experiment was to develop a lightweight model we have implemented a soft attention [28] in our model. Soft attention is an attention mechanism where the model calculates a weighted sum of all the elements in the input image. The "soft" aspect refers to the fact that the attention distribution is continuous and differentiable, meaning the model can focus on multiple parts of the input to varying degrees rather than selecting just one part. Here the soft attention unit accepts a feature tensor T of size rcd generated by the output Om from Eqn. (1) and applies convolution on T by employing a 3D kernel  of size 3×3×d which produces a feature map Fatt of size rc×1. Thus for n such kernels, feature maps of size rcn get generated. These feature maps undergo normalization using the softmax function and merging to determine a soft attention score ![image](https://github.com/user-attachments/assets/6d9db138-b171-4bd9-bf3c-d780dadf0317)
 given by

![Screenshot 2024-11-09 045503](https://github.com/user-attachments/assets/b3f32518-7799-479b-982d-526253287470)

The final weighted feature outcome is given by ![image](https://github.com/user-attachments/assets/87dad193-f0ff-4883-a9ed-85742ce4375a)
 in Eqn. (3) where the features in Fatt is multiplied with ![image](https://github.com/user-attachments/assets/9df3be1b-0faf-4637-85a1-559e57ec2d84)
 to enhance the significance of the relevant feature values. This product is re-scaled by a learning weight . Finally, the re-scaled weighted features are added with the original feature of ![image](https://github.com/user-attachments/assets/44ce5f4d-d626-4590-ba0b-d2c3eff856f6)
 to aid in performing optimal selection of feature map regions that are more relevant in the identification of a lung cancer class.

![image](https://github.com/user-attachments/assets/8bc735fb-ed63-4ae3-b107-55b8b1dd266e)

4.1 Experimental Setup
All the experimentation has been performed on Kaggle Notebook editor. We have utilized Kaggle’s GPU T4-X2 as an accelerator. Additionally for all the datasets we have ensured the same hyperparameters values as listed in Table 4. We preprocess lung cancer images from each dataset.

![image](https://github.com/user-attachments/assets/3d9fcfe7-0c2b-4e01-a728-add58748d896)




Result
![image](https://github.com/user-attachments/assets/6a8134dc-d986-4728-a584-8e3214d74246)

