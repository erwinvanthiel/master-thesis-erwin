# Thesis log

## Week 1

#### goals
- Build AC-WGAN_GP
- Train it on MNIST
- Generate some nice MNIST samples

#### Results
![ACGAN](/images/AC-WGAN_GP.png)

#### Conclusion
I was able to successfully implment AC-WGAN_GP and generate realistic MNIST samples with it.


## Week 2

#### goals
- Transfer GAN to generate adversarial examples

#### approach 1
Take a pre-trained discriminator, a pre-trained auxiliary classifier and a target classifier and train generator according to the losses composition depicted in the figure below.

![synthetic-adversarial-framework](/images/synthetic-adversarial-framework.png)

#### results
An untrained generator appeared to be unable to catch up with a fully trained discriminator. Consequently the generator samples remained just noise images.

#### approach 2
The second approach consisted of transfering a pre-trained generator and retraining it to produce adversarial examples. This setup strongly resembles that of [^AT-GAN]. The distance metric however was implemented by FID distance rather than euclidean distance. The results however were not as they were hoped to be. The figure below shows the degration of the appearance of the examples. 

![synthetic-adversarial-framework-with-fid](/images/synthetic-adversarial-framework-with-fid.png)

#### Conclusion
Thus far I have not been able to find a novel way to generate synthetic adversarial examples. To be continued

## Week 3

#### goals
- Attack multi-label classifier from ASL (https://github.com/Alibaba-MIIL/ASL) with single label attack methods
- Evaluate how well these methods work

#### Attacking a single label of a multi-label classifier
An implementation of PGD was used on the mlc-model but the attack had not the desired effect. After the attack was performed almost all positive labels were turned of in the prediction. 

## Week 4
In order to verify the working of the PGD implementation, the attack is performed in the single-label case. When a imageNet pretrained resnet model is attacked with pgd in MSCOCO the results show  that the attack has little influence, whereas the attack is very effective in on a single ImageNet image. 

## Week 5 

#### Goals
- Use PGD attack on ASL and show ineffectiveness. 
- start draw up of research question and discuss related literature.


#### Results
![pgd-attack-on-asl](/images/pgd-attack-on-asl.png)  


## Bibliography
[^AT-GAN]: Wang, X., He, K., & Hopcroft, J. E. (2019). AT-GAN: A generative attack model for adversarial transferring on generative adversarial nets. CoRR, abs/1904.07793.
