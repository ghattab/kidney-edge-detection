## Method

![](/figures/net.png){: .center-image }
> Modified U-Net Architecture. Modified with 3 convolutional layers on each level apart from the bottleneck. After each max pooling layer (red) a dropout layer (dark gray) was added. Each dropout layer has a dropout value set to $0.1$ except the first one, it is set to $0.05$. The blue layers in combination with the following convolutional layers depict transpose convolutional layers. Arrows represent skip connections. Concatenation is done after the transpose convolution.

## Results


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
