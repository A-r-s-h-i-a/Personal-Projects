# Progressive Growing of GANs (ProGAN)
NVIDIA's ProGAN technique was tested here on generating low-quality human faces. The associated paper can be found in this directory, along with the "celebHQ" dataset which contains the faces used for training. The idea behind ProGANs is to purposefully start from a point of low performance, and grow both the generator and discriminator progressively. As they are being trained/improving, additional layers are also being added to each. This is intended to not only speed up training, but also to stabilize it.

As we can see from some of the outputs of the trained Generator, this technique is quite effective. Especially as I did not carry it out with the resources that NVIDIA did, but still recieved quality results.

# Results
![ProGAN](https://github.com/A-r-s-h-i-a/Personal-Projects/blob/main/ProGAN/32x32-1.png)
![ProGAN](https://github.com/A-r-s-h-i-a/Personal-Projects/blob/main/ProGAN/32x32-2.png)
![ProGAN](https://github.com/A-r-s-h-i-a/Personal-Projects/blob/main/ProGAN/individualImage.png)
