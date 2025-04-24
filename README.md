 This repository contains code and resources that were utilized in the following paper:



> ***Lung Cancer Identification from CT Scans using a Soft-attention enabled Deep Transfer Learning Model***\
> *S. Dev, P. S. Roy, N. Chakraborty and R. Sarkar*\
> Published in: IEEE ISACC 2025\
> Paper: https://doi.org/10.1109/ISACC65211.2025.10969319


If you use this code or find it helpful, please consider citing the paper.
```bibtex
@inproceedings{10969319,
  author    = {S. Dev, P. S. Roy, N. Chakraborty and R. Sarkar},
  title     = {Lung Cancer Identification from CT Scans using a Soft-attention enabled Deep Transfer Learning Model},
  booktitle = {2025 3rd International Conference on Intelligent Systems, Advanced Computing and Communication (ISACC)},
  pages     = {254-259},
  year      = {2025},
  doi       = {10.1109/ISACC65211.2025.10969319}
}
```
You can download the paper here as well [IEEE ISACC 2025 10969319](https://github.com/user-attachments/files/19899861/IEEE.ISACC.2025.Lung_Cancer_Identification_from_CT_Scans_using_a_Soft-attention_enabled_Deep_Transfer_Learning_Model.pdf)

<!-- [This content will not appear in the rendered Markdown](https://docs.google.com/viewer?url= ) -->
----------------

Sample images from datasets, diagnosed as either lung cancer or healthy.

![image](https://github.com/user-attachments/assets/ba384f60-3329-4eb5-9fb0-4901aac36e97)





	


# Datasets used
In this research, three popular datasets which are publicly available, are considered for training and evaluating the proposed system. These datasets are relatively small and provide the necessary challenges that tend to pose limitations of the DL models. The details of the considered datasets are provided in the following sub-sections:
## IQ-OTH/NCCD
   
The dataset comprises CT scans of patients at various stages of lung cancer and healthy subjects, marked by oncologists and radiologists. The dataset consists of 1190 CT scan images from 110 cases, categorized into three classes: normal (55 cases), benign (15 cases), and malignant (40 cases). Table 1 shows the organization of the IQ-OTH/NCCD dataset.

![Screenshot 2024-11-09 040000](https://github.com/user-attachments/assets/8349f2e3-ba33-41b4-a846-8a30d701c642)

## LC25000

This dataset consists of 25,000 histopathological images, divided into five distinct classes. We consider only three classes namely, Lung Benign tissue, Lung Adenocarcinoma and Lung Squamous cell carcinoma. As depicted in Table 2, each class contains ≈ 5000 images divided into 80% training set while validation and test sets having 10% images each. All images are 768×768 pixels in size and are in jpeg file format. The dataset contains histopathological images of lung and colon tissues. Since our work is centered around lung cancer, we would only be considering the lung image set comprising 15000 images.

![Screenshot 2024-11-09 040304](https://github.com/user-attachments/assets/73b89f4e-05e9-4372-8c73-30ccf6ebe41a)

## LIDC-IDRI

This dataset consists of diagnostic and lung cancer screening thoracic CT scans with marked-up annotated lesions. It is a web-accessible international resource for development, training, and evaluation of CAD methods for lung cancer detection and diagnosis.The dataset consists of 2066 images distributed across a train set of 1323 images, a test set of 413 images and a validation set of 330 images. Each set further has data distributed across 3 subclasses viz., benign and malignant. This dataset comprises diagnostic and lung cancer screening thoracic CT scans with annotated lesions.

![image](https://github.com/user-attachments/assets/48ddae7c-8310-40f2-b57f-ca9fa84d2c94)


# Pre-trained DL Models
Here, two base models called MobileNetV2 and DenseNet121 are explored for understanding the efficiency in lung cancer identification. In both the cases, an input image of size 224×224×3 from the lung cancer image dataset 

## MobilenetV2

In MobileNetV2, there are two types of blocks. As shown in Fig. 3, one is residual block with stride of and Another one is block with stride of 2 for downsizing. There are 3 layers for both types of blocks. This time, the first layer is 1×1 convolution with ReLU6. The second layer is the depthwise convolution. The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if ReLU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain. And there is an expansion factor t. And t=6 for all main experiments. If the input got 64 channels, the internal output would get 64×t=64×6=384 channels where t: expansion factor, c: number of output channels, n: repeating number, s: stride. 3×3 kernels are used for spatial convolution. In typical use, the primary network (width multiplier 1, 224×224), has a computational cost of 300 million multiply-adds and uses 3.4 million parameters. The performance tradeoffs are further explored, for input resolutions from 96 to 224, and width multipliers of 0.35 to 1.4. The network computational cost up to 585M MAdds, while the model size varies between 1.7M and 6.9M parameters. To train the network, 16 GPUs are used with a batch size of 96.

![image](https://github.com/user-attachments/assets/3bedb966-2c5d-4c2f-b4bc-5ffc5239e5bc)

## DenseNet121

In DenseNet121, there are two types of layers: convolutional layers within dense blocks and transition layers for down-sampling as depicted in Fig. 4. Each dense block consists of multiple layers where each layer receives input from all preceding layers. There are 3 key operations in each dense block layer. The first is a 1×1 convolution followed by ReLU. The second is a 3×3 convolution, and the third operation is concatenation of the input with its preceding layers’ outputs, promoting feature reuse. There is a growth rate k that determines the number of output channels per layer. For all main experiments, k is typically set to 32. If the input has 64 channels, after one layer, the output would have 64 + k = 64 + 32 = 96 channels. Transition layers, which include a 1×1 convolution followed by 2×2 average pooling, are used for down-sampling between dense blocks. These layers help control the complexity of the network and manage the size of the feature maps.
![image](https://github.com/user-attachments/assets/4c89e2a9-b02e-460b-ab8b-8f16e000e521)

# Soft Attention Scheme
As the primary aim of our experiment was to develop a lightweight model we have implemented a soft attention [28] in our model. Soft attention is an attention mechanism where the model calculates a weighted sum of all the elements in the input image. The "soft" aspect refers to the fact that the attention distribution is continuous and differentiable, meaning the model can focus on multiple parts of the input to varying degrees rather than selecting just one part. Here the soft attention unit accepts a feature tensor T of size rcd generated by the output Om from Eqn. (1) and applies convolution on T by employing a 3D kernel  of size 3×3×d which produces a feature map Fatt of size rc×1. Thus for n such kernels, feature maps of size rcn get generated. These feature maps undergo normalization using the softmax function and merging to determine a soft attention score ![image](https://github.com/user-attachments/assets/6d9db138-b171-4bd9-bf3c-d780dadf0317)
 given by

![Screenshot 2024-11-09 045503](https://github.com/user-attachments/assets/b3f32518-7799-479b-982d-526253287470)

The final weighted feature outcome is given by ![image](https://github.com/user-attachments/assets/87dad193-f0ff-4883-a9ed-85742ce4375a)
 in Eqn. (3) where the features in Fatt is multiplied with ![image](https://github.com/user-attachments/assets/9df3be1b-0faf-4637-85a1-559e57ec2d84)
 to enhance the significance of the relevant feature values. This product is re-scaled by a learning weight . Finally, the re-scaled weighted features are added with the original feature of ![image](https://github.com/user-attachments/assets/44ce5f4d-d626-4590-ba0b-d2c3eff856f6)
 to aid in performing optimal selection of feature map regions that are more relevant in the identification of a lung cancer class.

![image](https://github.com/user-attachments/assets/8bc735fb-ed63-4ae3-b107-55b8b1dd266e)

## An overview of the proposed system.

![image](https://github.com/user-attachments/assets/585a5da0-dc18-40d2-a78c-6f5f8747b0ec)


# Experimental Setup
All the experimentation has been performed on Kaggle Notebook editor. We have utilized Kaggle’s GPU T4-X2 as an accelerator. Additionally for all the datasets we have ensured the same hyperparameters values as listed in Table 4. We preprocess lung cancer images from each dataset.

![image](https://github.com/user-attachments/assets/3d9fcfe7-0c2b-4e01-a728-add58748d896)




# Result

## For IQOTH

### For Densenet121

![Screenshot 2024-11-09 044155](https://github.com/user-attachments/assets/ee87807e-2f7c-414f-999a-e11755bac3a4)

Test Loss: 0.03285326808691025

Test Accuracy: 0.9954545497894287

#### Confusion Matrix

![f89598ae-09b4-4abc-9c12-18ed119d1596](https://github.com/user-attachments/assets/ec3c5b8a-0259-4aec-ac1e-5971bcc96b2d)

#### Test Accuracy vs Epochs

![bbefaf63-22f0-4e3f-ae1e-29d53d1dd8c8](https://github.com/user-attachments/assets/96104570-5311-4ed2-a236-3fc849bfd8b6)

#### Loss Curve

![4e0e7d81-5238-4640-aed4-93af842cc040](https://github.com/user-attachments/assets/c2326855-03b9-4332-853a-476b34eff4cf)

### For MobileNetv2

![Screenshot 2024-11-09 044304](https://github.com/user-attachments/assets/58cd7a16-42c2-4658-b2aa-2f3be4ddb463)

Test Loss: 0.06389608979225159

Test Accuracy: 0.9863636493682861

#### Confusion Matrix

![305119ce-0fa6-4e19-ab71-dca2dff8c90d](https://github.com/user-attachments/assets/66551d1d-76e0-41d2-a5df-7be985870ebc)

#### Test Accuracy vs Epochs

![3cb91f12-4b81-40f5-8bd4-4eea12f4f978](https://github.com/user-attachments/assets/e2c4a389-c9d6-4ea4-be3e-18e87c877a37)

#### Loss Curve

![4ad23f52-b902-4ba4-9f83-a3834fe0cd37](https://github.com/user-attachments/assets/37ef3333-7916-491a-b2b4-85f0e87224df)


## For LC25000

### For DenseNet121

![Screenshot 2024-11-09 043633](https://github.com/user-attachments/assets/4f1d45a4-ecb9-4ad4-925e-b34cc6163a41)

Test Loss: 0.2601519525051117

Test Accuracy: 0.9673333168029785

#### Confusion Matrix

![6c7a7342-0362-4b1a-8bcc-50bf9fd33615](https://github.com/user-attachments/assets/4a2e1524-39f7-4a37-bb86-1445939f1f58)

#### Test Accuracy vs Epochs

![ec7fca03-f752-4c6a-9228-d9ac1d80727e](https://github.com/user-attachments/assets/0426fd3a-f936-495e-bcb2-60e5bc74e834)

#### Loss Curve

![95b36237-e8ea-42bf-b826-7a8532e93581](https://github.com/user-attachments/assets/2aa08229-a623-43f8-95d9-c3c9499d1f98)

### For MobileNetv2

![Screenshot 2024-11-09 043525](https://github.com/user-attachments/assets/8d39ec40-0dab-4dc0-a19a-8ff6bcc68ce5)

Test Loss: 0.1851891279220581

Test Accuracy: 0.968666672706604

#### Confusion Matrix

![bf547ed1-9b26-4340-b05f-0ae718d3176f](https://github.com/user-attachments/assets/a479363e-ebf4-4d42-a103-e2fc27f2ee40)


#### Test Accuracy vs Epochs

![cf2c0010-f383-4035-9d56-0d41995703c5](https://github.com/user-attachments/assets/5da7a15f-abba-452f-9a73-b25b5e7211e7)

#### Loss Curve

![15d0027c-c414-4901-bab9-6b68a7c13648](https://github.com/user-attachments/assets/612b5103-ca6b-4113-be9d-efc4e0018ea4)


## For LIDC

### For Densenet121

![Screenshot 2024-11-09 043804](https://github.com/user-attachments/assets/f3aa5421-9bac-4519-9d78-0b35cc4ae35c)

Test Loss: 0.11375235766172409

Test Accuracy: 0.9491525292396545

#### Confusion Matrix

![eed2519d-482e-4e83-8d2d-aac23f0464cd](https://github.com/user-attachments/assets/0759a4bb-acaf-4653-9600-772a8a82be15)

#### Test Accuracy vs Epochs

![9063aee9-11d3-4a47-9963-d6eae9494b3b](https://github.com/user-attachments/assets/cd02b8b5-a1dc-49f8-bf90-896b41492d30)

#### Loss Curve

![31af7817-79a9-491a-a83f-cc021ffd4f0b](https://github.com/user-attachments/assets/d25bba32-2fd4-4a9f-8829-4902053dac33)

### For MobileNetv2

![Screenshot 2024-11-09 043904](https://github.com/user-attachments/assets/04f4b9e7-055c-4f22-bab4-9d15510819e9)

Test Loss: 0.28500422835350037

Test Accuracy: 0.9200968742370605

#### Confusion Matrix

![a59614d6-98cd-4b58-9322-8625c9ad34ee](https://github.com/user-attachments/assets/0c12a5d7-f65d-4987-8fce-b4329c9df12d)

#### Test Accuracy vs Epochs

![f497f662-4afb-4c79-9291-32dc1b242793](https://github.com/user-attachments/assets/4bb7c9b0-c5ca-4503-b5a7-84df26234af0)

#### Loss Curve

![794bfe62-e017-478a-8a2b-31ab50f9de05](https://github.com/user-attachments/assets/63d52fd5-697a-47c4-a0fa-70cca41c0cf7)


