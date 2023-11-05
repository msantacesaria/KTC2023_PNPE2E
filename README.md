# Code for KTC2023 EIT challenge (end-to-end + PnP)


## Brief description of the algorithm
This repository contains the Python files required by the organisers of the EIT Kuopio challenge 2023 which implements an end-to-end approach followed by a PnP reconstruction algorithm which makes use of the segmented images computed by the first part of the algorithm to improve reconstruction.

For the end-to-end approach, training was performed using GT simulated data with random polygons of different shapes, size and postive/negative contrast. The end-to-end procedure includes a step where a pseudo-inverse operator is also learned. For each simulated inclusion the corresponding segmentation is matched with the one computed directly on the GT data.

For the Plug and Play part, training was performed by associating GT data with their noisy versions which were computed running a regularised Gauss-Newton algorithm (on meshes) for few (5) iterations on the corresponding measurements simulated using the forward model provided.

Training was performed using GT simulated data with random polygons of different shapes, size and postive/negative contrast. In more detail, we created synthetic shapes of:
- rectangles with sides of different length and with different orientation within the domain
- squares with side of different length and with different orientation within the domain
- generic triangles with sides of different lenghts
- generic quadrilaterals with sides of different lengthts
- circles of different radii and centres
- horseshoe shapes of different sides and orientations
- star-like shapes with 3-4-5 tips with different centres and orientations
For rectangles, squares, circles and star-like shapes we further allowed the possibility of having holes of different size.
All shapes were assigned to a random negative/positive value within fixed ranges to simulate resistive/conductive inclusions. We allowed a number of inclusions shaped as above equal to 1, 2, 3, making sure that all shapes do not intersect and are note placed too close to the boundary.

Here are some simulated inclusions (different shapes, size, number, position):
![Fig2](https://github.com/lucala00/KTC2023_E2E/assets/49308207/7143a902-d650-4c2e-bfe6-da21a19a9550)
![fig5](https://github.com/lucala00/KTC2023_E2E/assets/49308207/ef9111ad-8e03-46dc-83ca-2f548cebebb3)
![Figure_3](https://github.com/lucala00/KTC2023_E2E/assets/49308207/1960ec95-f80b-4b35-b3a7-6d8e7ed69e1c)
![fig1](https://github.com/lucala00/KTC2023_E2E/assets/49308207/c034634f-363c-4c60-99fc-a8aa7ae59a43)

Associated noisy versions were computed running a proximal Gauss-Newton method (on meshes) with a pre-trained graph neural denoiser in place of the standard proximal map for few (5) iterations on the corresponding measurements simulated using the forward model provided.

To improve the segmentation performed on the reconstructions obtained, we incorporated a post-processing step based on artificial masks given by an end-to-end approach (see [here](https://github.com/lucala00/KTC2023_E2E) fore more details). Here are some examples of the masks for each level of difficulty.

![e2e](https://github.com/lucala00/KTC2023_E2E/assets/49308207/089da0f9-0e6d-45ea-9bcc-6aea5ca01884)

These masks were superimposed on the final segmented reconstruction given by the PnP approach, i.e. the pixels belonging to the background of the masks were set to 0.

For the PnP algorithm, we considered a graph-U-net denoiser extending the CNN-based denoiser to non-Euclidian manifold domain. It relies on the Graph U-net architecture, a U-Net-like architecture for graph data which allows high-level feature encoding and decoding for network embedding. It is based on a convolution graph kernel and gPool and gUnpool layers. The pool (gPool) operation samples some nodes to form a smaller graph based on their scalar projection values on a trainable projection vector. As an inverse operation of gPool, the unpooling (gUnpool) operation restores the graph to its original structure with the help of locations of nodes selected in the corresponding gPool layer.

The GU-Net-denoiser as well as the Graph-U-Net can be formalized as a composition of layers, where each layer is characterized by the composition of a graph convolution, which is nothing but a ReLU activation function σ and a gPool/gUnpool operator

## Authors:
- Tatiana Bubba, University of Bath, UK, tab73 AT bath.ac.uk
- Luca Calatroni, CNRS, FR, calatroni AT i3s.unice.fr
- Damiana Lazzaro, University of Bologna, IT, damiana.lazzaro AT unibo.it 
- Serena Morigi, University of Bologna, IT, serena.morigi AT unibo.it 
- Luca Ratti, University of Bologna, IT, luca.ratti5 AT unibo.it
- Matteo Santacesaria, University of Genoa, IT, matteo.santacesaria AT unige.it 
- Julian Tachella, CNRS, FR, julian.tachella AT ens-lyon.fr

## Installation instructions and requirements

Please use the following command to install the required packages

```pip install -r requirements.txt```

Note that CUDA is required to run parts of the code. Also some newer version of the packages might give worse results (in particular torch-geometric should be version 2.3.1). We also make use of the library [deepinv](https://github.com/deepinv/deepinv). To install the latest stable release of deepinv, you can simply do:

```pip install deepinv```

You can also install the latest version of deepinv directly from github:

``` pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv``` 

We created a script main.py to reconstruct the inclusions provided for training from voltage measurements:

```python main.py /path_to_input_folder /path_to_ouput_folder difficulty_level```

The same architecture was employed for all levels of difficulties. Training was repeated to better adapt to each level of difficulty. For each level of difficulty, the learned parameters of the network are available in at the path

```/models/difficulty_{j}_learnedlinear.pth.tar ```

and uploaded directly via the call to the main file once the difficulty level is specified.

For the denoiser, the learned weights are stored in the file
``` weights_denoiser.pth```

## Examples

![Examples_PnP_E2E_merge](https://github.com/msantacesaria/KTC2023_PNPE2E/assets/148894828/76c12696-5ff6-4d3d-84ff-50d7b3b84e51)


## References

* Samuel Hurault, Mathieu Terris, Julian Tachella, DeepInverse: a Pytorch library for imaging with deep learning, https://deepinv.github.io.
*  Francesco Colibazzi, Damiana Lazzaro, Serena Morigi, Andrea Samoré. Deep-plug-and-play proximal Gauss-Newton method with applications to nonlinear, ill-posed inverse problems. Inverse Problems and Imaging, 2023, 17(6): 1226-1248. doi: 10.3934/ipi.2023014
* Francesco Colibazzi, Damiana Lazzaro, Serena Morigi, Andrea Samoré. Learning Nonlinear Electrical Impedance Tomography. J Sci Comput, 2022, 90(58). https://doi.org/10.1007/s10915-021-01716-4
