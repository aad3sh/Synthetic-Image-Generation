# Synthetic-Image-Generation
Generating fake images from noise using deep convolutional GANs
- The model consisted of a generator network fed with gaussian noise, that produces an image from it, and a discriminator network that predicts whether the image generated is fake or true. 
- The GAN is then compiled and loss from the discrimator was fed into the generator to re-adjust it’s weights till it produces a synthetic image realistic enough to fool the discriminator. 
