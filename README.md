# FCOS
### Model trained utilizing the VOC2007DetectionTiny dataset ###

This is a recreation of the FCOS model outlined in the paper cited below. There is one key difference between their model and my model. In the paper outlined below there are two
stems used to produce the classification logits, and box deltas. The classification logits are then used (in my model) to predict ONLY the class of the object. The box deltas created by the second stem are used to predict the bounding box, and centerness of the bounding box. 

This key difference follows a more intuitive approach and should yield better results. I could not tests due to Google Colab payment limits; however, upon more literature review in practice this is true.

Anyhow, I will describe the entirety of the model in short and display the loss graph over 9000 iterations trained in about 2-3 hours on VOC2007Detection Tiny. Then I will list some ways I think this model could achieve greater performance on mAP (mean average precision).

The model is made of two different models. It is a two-staged object detector.
Firstly, it utilizes a backbone model of three 1x1 convolutions. These are called C3,C4,C5. 

The model utilizes a feature pyramid to calculate loss at each respective level and minimize. The feature pyramid utilizes 3x3 convolutions to extract the feature maps at each level, these levels are referred to as either FPN (feature pyramid level) or p3,p4,p5. In the feature pyramid to ensure features are shared to lower levels we interpolate higher up convolutions to lower convolutions (higher and lower refers to order of execution, higher being later). Any type of interpolation should work sufficiently well, however, bilinear is probably best (as for now). We refer to these levels as p5, p4p5, and p3p4p5 (indicating the interpolation). p3p4p5 is not a simple 3x3 convolution as the other heads, but includes batch normalization, such that it follows (Conv, BN, ReLU, Conv). At each level there is a 'head', this head calculates the box regression loss (l1 loss), the centerness loss (binary cross entropy loss with logits specifically), and (utilizing the separate head) the classification loss (sigmoid focal loss). 

If it wasn't apparent the name FCOS is very fitting.

During inference, the model is capable of near real time classification and utilizes NMS (Non-Maximum Suppression) to remove poor object predictions.

Loss Curve:
![image](https://github.com/Siggyv/FCOS_VOC2007DetectionTiny/assets/93465187/5099a29f-8d2c-43f2-9b5a-994f5a7e8fcb)

Inference Image:
![image](https://github.com/Siggyv/FCOS_VOC2007DetectionTiny/assets/93465187/6750c703-b2d0-4ec9-9b75-19915563944b)

Inference took 1.2 seconds for 4 images.

mAP is 22%

Potential improvements:
- As common in AI, I believe a larger dataset would help the model perform much better. More akin to the performance in the paper below (this includes more training time).
- A larger backbone model would allow the model to learn more complex features, the paper utilizes resNet-50, which I utilize a simple 3 layer CNN.
- The code written does not do enough attention to detail to maintain device consistency, additionally, torch broadcasting and tensor operations could potentially be applied to this network to speed up training time. (Maybe concurrency, but would avoid this due to encapsulation).
- I believe there may be an issue with switching x, and y values, but it is hard to tell where this occurs.

Zhi Tian, Chunhua Shen, Hao Chen, and Tong He, "FCOS: Fully Convolutional One-Stage Object Detection," 2019, arXiv:1904.01355.
