# Few-Shot-Image-Generation-Using-VAE

The aim of this project is to create novel and creative images using few-shot image 
generation. This project will provide assistance in creative process for different artists. Artists or 
designers who lack time or creative inspiration for multiple versions of an image could sketch a 
limited number of drawings. They can have the trained few-shot learning model, generate 
multiple similar versions of the sketches that they produced. 
Earlier, Generative Adversarial Networks (GAN) were used to generate realistic images. 
However, GAN’s require inordinate amount of data. Meta-learning can be used to bypass this 
hurdle of less data. Hence the goal of our project is to use Few-shot Image Generation by 
manipulating latent features of generative models. 

#Approach 
In order to address the above mentioned goal, we took the following two routes: 
1. Variational Auto-Encoder: VAE are a type of generative models. They are based off of 
auto-encoders. Auto encoders have two parts, encoders and decoders. VAE learn to 
generate new data by minimizing the reconstruction loss and latent loss. What goes in to 
the network is spit out making sure there is as little difference as possible. It is also made 
sure that the latent vector takes only specific set of values. 
2. Reptile Algorithm to Meta train the model : Reptile seeks an initialization for the 
parameters of a neural network, such that the network can be fine-tuned using a small 
amount of data from a new task. Reptile simply performs stochastic gradient descent 
(SGD) on each task in a standard way. 


#Dataset
MNIST 
OMNIGLOT 
FIGR-8 

#Training
python train.py
 
