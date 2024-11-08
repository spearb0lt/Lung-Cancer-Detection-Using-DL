# LC25000

This dataset consists of 25,000 histopathological images, divided into five distinct classes. We consider only three classes namely, Lung Benign tissue, Lung Adenocarcinoma and Lung Squamous cell carcinoma. As depicted in Table 2, each class contains ≈ 5000 images divided into 80% training set while validation and test sets having 10% images each. All images are 768×768 pixels in size and are in jpeg file format. The dataset contains histopathological images of lung and colon tissues. Since our work is centered around lung cancer, we would only be considering the lung image set comprising 15000 images.

![Screenshot 2024-11-09 040304](https://github.com/user-attachments/assets/73b89f4e-05e9-4372-8c73-30ccf6ebe41a)

# Performance on LC25000

## For DenseNet121

![Screenshot 2024-11-09 043633](https://github.com/user-attachments/assets/4f1d45a4-ecb9-4ad4-925e-b34cc6163a41)

Test Loss: 0.2601519525051117

Test Accuracy: 0.9673333168029785

### Confusion Matrix

![6c7a7342-0362-4b1a-8bcc-50bf9fd33615](https://github.com/user-attachments/assets/4a2e1524-39f7-4a37-bb86-1445939f1f58)

### Test Accuracy vs Epochs

![ec7fca03-f752-4c6a-9228-d9ac1d80727e](https://github.com/user-attachments/assets/0426fd3a-f936-495e-bcb2-60e5bc74e834)

### Loss Curve

![95b36237-e8ea-42bf-b826-7a8532e93581](https://github.com/user-attachments/assets/2aa08229-a623-43f8-95d9-c3c9499d1f98)

## For MobileNetv2

![Screenshot 2024-11-09 043525](https://github.com/user-attachments/assets/8d39ec40-0dab-4dc0-a19a-8ff6bcc68ce5)

Test Loss: 0.1851891279220581

Test Accuracy: 0.968666672706604

### Confusion Matrix

![bf547ed1-9b26-4340-b05f-0ae718d3176f](https://github.com/user-attachments/assets/a479363e-ebf4-4d42-a103-e2fc27f2ee40)


### Test Accuracy vs Epochs

![cf2c0010-f383-4035-9d56-0d41995703c5](https://github.com/user-attachments/assets/5da7a15f-abba-452f-9a73-b25b5e7211e7)

### Loss Curve

![15d0027c-c414-4901-bab9-6b68a7c13648](https://github.com/user-attachments/assets/612b5103-ca6b-4113-be9d-efc4e0018ea4)


# You Can View My Notebook Below

https://docs.google.com/viewer?url=https://github.com/user-attachments/files/17684858/LC25000-main.pdf
