## Evaluation

### Quantitative
Please refer to [EVAL.md](https://github.com/ghattab/kidney-edge-detection/blob/master/EVAL.md) for the full evaluation results of the leave one out cross validation using our method, and the vanilla U-net implementation. A comparison to the 2017 challenge winner is also reported. Raw data can be found under the analysis folder.

### Qualitative
Qualitative examples are reported under [figures](https://github.com/ghattab/kidney-edge-detection/tree/master/figures).


* `input.png` Example RGBD input for frame 100 of data set 1. Left: original left image center: disparity, and right: RGB image.
* `challenge.png` Example challenges in kidney boundary detection. Left image is from the left camera and right image is the same image with an overlay of the inverted signed-distance the reference boundary (purple) and the predicted boundary (teal). Top row shows two frames from data set 18 and bottom row shows two others from data set 16.
* `output.png` Example output for frame 100 of data set 1. Left: raw network output and right: postprocessed image.

## Network

![](/figures/net.png)
> Kidney Boundary Network (KiBo-Net) Architecture. Modified with 3 convolutional layers on each level apart from the bottleneck. After each max pooling layer (red) a dropout layer (dark gray) was added. Each dropout layer has a dropout value set to 0.1 except the first one, it is set to 0.05. The blue layers in combination with the following convolutional layers depict transpose convolutional layers. Arrows represent skip connections. Concatenation is done after the transpose convolution.

### Random seeds

The networks where trained with the following random seeds. To reproduce our results use the ``--seeds`` argument of ``train.py``.

<table>
<tr><td>

| LOOCV |    Seed   |
|-------|:---------:|
|   1   | 211338299 |
|   2   |  66992261 |
|   3   | 612395054 |
|   4   |  29634478 |
|   5   |  82962696 |
|   6   |  12737697 |
|   7   |  37241791 |
|   8   |   133833  |


</td><td>

| LOOCV |    Seed   |
|-------|:---------:|
|   9   |  4908350  |
|   10  |  8416991  |
|   11  | 957795873 |
|   12  | 572028758 |
|   13  | 750409111 |
|   14  |  66171044 |
|   15  |  27924390 |
|   16  | 233707090 |

</td></tr> </table>

# Running the KiBo-Net

The easiest way to inference images using the KiBo-Net is using our Docker container located at: https://hub.docker.com/r/fuxxel/kibo-net

The Docker container already contains an example data set (test set 18). 
To process the example data set execute the following commands:
```
# Pull the docker image 
docker pull fuxxel/kibo-net:gpu

# use nvidia-docker to run the network using the GPU (interactively)
nvidia-docker run -it fuxxel/kibo-net:gpu bash 
```
Inside the docker container navigate to the kibo subdirectory:
```
cd kibo
ll 
```
The directory contains all necessary scripts and test set 18 inside the folder sample_input.

To start the whole processing pipeline run:
```
./run_pipeline.sh
```
The final results will be inside the sample_input/network_output directory.

## Running on custom data

To use your own data you can replace the content of sample_input/left_frames and sample_input/right_frames with your data.
Additionally, you need to provide a camera_calibration.txt file and place it into sample_input.

The following example demonstrates which camera parameters **must** be defined:
```
Camera-0-F: 1084.21 1084.05                               // left camera x,y focal dist in pixels
Camera-0-C: 580.02 506.79                                 // left camera x,y center in pixels
Camera-0-K: -0.00069 0.00195 0.00018 0.00000 0.00000      // left camera radial distortion
Camera-1-F: 1083.43 1083.22                               // right camera x,y focal dist in pixels
Camera-1-C: 680.91 505.81                                 // right camera x,y center in pixels
Camera-1-K: -0.00085 0.00245 0.00004 0.00000 0.00000      // right camera radial distortion
Extrinsic-Omega: -0.0002 -0.0011 -0.0000                  // left to right camera rotation
Extrinsic-T: -4.3499 0.0333 -0.0369                       // left to right camera position
```

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
