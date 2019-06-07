## Evaluation

### Quantitative
Please refer to [EVAL.md](https://github.com/ghattab/kidney-edge-detection/blob/master/EVAL.md) for the full evaluation results of the leave one out cross validation using our method, and the vanilla U-net implementation. A comparison to the 2017 challenge winner is also reported.

### Qualitative
| ![challenge](https://raw.githubusercontent.com/ghattab/kidney-edge-detection/master/figures/challenge.png?token=ADJ6FUJBQMIGCBTBTTONGUK5ANQHM) | 
|:--:| 
| Example challenges in kidney boundary detection. Left image is from the left camera and right image is the same image with an overlay of the inverted signed-distance the reference boundary (purple) and the predicted boundary (teal). Top row shows two frames from data set 18 and bottom row shows two others from data set 16. |


## Network

![](/figures/net.png)
> Modified U-Net Architecture. Modified with 3 convolutional layers on each level apart from the bottleneck. After each max pooling layer (red) a dropout layer (dark gray) was added. Each dropout layer has a dropout value set to 0.1 except the first one, it is set to 0.05. The blue layers in combination with the following convolutional layers depict transpose convolutional layers. Arrows represent skip connections. Concatenation is done after the transpose convolution.

### Input

| ![input](https://raw.githubusercontent.com/ghattab/kidney-edge-detection/master/figures/input.png?token=ADJ6FUMYWPTSZ6T2RPQFS5C5ANQHQ) | 
|:--:| 
| Example RGBD input for frame 100 of data set 1. Left: original left image center: RGBD input image and right: disparity.|

### Output

| ![output](https://raw.githubusercontent.com/ghattab/kidney-edge-detection/master/figures/output.png?token=ADJ6FUKQOJVJFUBMPETQN5C5ANQHY) | 
|:--:| 
| Example output for frame 100 of data set 1. Left: raw network output and right: postprocessed image.|


## License
This work is available under an Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0).
This is a human-readable summary of (and not a substitute for) the license.
Disclaimer.
You are free to:
```
    Share — copy and redistribute the material in any medium or format
    Adapt — remix, transform, and build upon the material

    The licensor cannot revoke these freedoms as long as you follow the license
terms.
```
Under the following terms:
```
    Attribution — You must give appropriate credit, provide a link to the
license, and indicate if changes were made. You may do so in any reasonable
manner, but not in any way that suggests the licensor endorses you or your use.

    NonCommercial — You may not use the material for commercial purposes.

    ShareAlike — If you remix, transform, or build upon the material, you must
distribute your contributions under the same license as the original.

    No additional restrictions — You may not apply legal terms or
technological measures that legally restrict others from doing anything the
license permits.
```
Notices:
```
    You do not have to comply with the license for elements of the material in
the public domain or where your use is permitted by an applicable exception or
limitation.
    No warranties are given. The license may not give you all of the permissions
necessary for your intended use. For example, other rights such as publicity,
privacy, or moral rights may limit how you use the material.
```
