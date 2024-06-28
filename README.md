# xxxx: Transformer-based Generative Adversarial network for 3D CT-to-MRI modality translation
This repository is an official implementation of the work named ""
# Introduction
This network adopts a dual-input generative adversarial network(GAN). 
It mainly consists of three parts:
the convolutional branch(High Resolution Part) contained in the Generator is dedicate to extract the local feature information, 
while the transformer branch(Low Resolution Part) contained in the Generator is focus on the extraction of global long-range dependencies. 
The Discriminator is a fully-convolutional network which is used to distinguish the realness of generated MRI volumes.

![Raw file](../main/Framework Overview.png)
