IDS_VaE
===
The IDS_VaE project develops a cutting-edge Intrusion Detection System (IDS) utilizing deep learning. By integrating Convolutional Neural Networks (CNNs) and self-attention mechanisms, this system significantly enhances the detection accuracy of network security systems. Following the preprocessing guidelines outlined by Yu et al., 2021 (as detailed in our final report), the project employs a Variational Autoencoder (VaE) model initially trained on the CIC-IDS-2018 dataset. We then adapt and transfer the encoder segment of this model to the CIC-IDS-2017 dataset. With only 20% of the data used for training, the classifier achieves high accuracy and F1 scores on the remaining 80% of the data, demonstrating the distinctive characteristics of the byte-level packet data distribution.

The IDS_VaE project also compares supervised and self-supervised learning methodologies to assess model generalizability when transferring across different datasets. Our results highlight that the structure of the Variational Autoencoder (VaE), which focuses on identifying latent features in data, outperforms direct prediction methods. This superior performance is attributed to the favorable distribution characteristics of the data. Additionally, the inherent randomness in the VaE approach enables the use of larger models, enhancing their capacity to capture and represent data features more effectively.
