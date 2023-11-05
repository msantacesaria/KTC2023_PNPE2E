# Code for KTC2023 EIT challenge (end-to-end + PnP)


## Brief description of the algorithm
This repository contains the Python files required by the organisers of the EIT Kuopio challenge 2023 which implements an end-to-end approach followed by a PnP reconstruction algorithm which makes use of the segmented images computed by the first part of the algorithm to improve reconstruction.

For the end-to-end approach, training was performed using GT simulated data with random polygons of different shapes, size and postive/negative contrast. The end-to-end procedure includes a step where a pseudo-inverse operator is also learned. For each simulated inclusion the corresponding segmentation is matched with the one computed directly on the GT data.

For the Plug and Play part, training was performed by associating GT data with their noisy versions which were computed running a regularised Gauss-Newton algorithm (on meshes) for few (5) iterations on the corresponding measurements simulated using the forward model provided.

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

Note that CUDA is required to run parts of the code. Also some newer version of the packages might give worse results (in particular torch_geometric should be version 2.3.1).

We created a script main.py to reconstruct the inclusions provided for training from voltage measurements:

```python main.py /path_to_input_folder /path_to_ouput_folder difficulty_level```

The same architecture was employed for all levels of difficulties. Training was repeated to better adapt to each level of difficulty. For each level of difficulty, the learned parameters of the network are available in at the path

```/models/difficulty_{j}_learnedlinear.pth.tar ```

and uploaded directly via the call to the main file once the difficulty level is specified.

To install the latest stable release of deepinv, you can simply do:

```pip install deepinv```

You can also install the latest version of deepinv directly from github:

``` pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv``` 

For the denoiser, the learned weights are stored in the file
``` weights_denoiser.pth```

## References

* Samuel Hurault, Mathieu Terris, Julian Tachella, DeepInverse: a Pytorch library for imaging with deep learning, https://deepinv.github.io.
*  Francesco Colibazzi, Damiana Lazzaro, Serena Morigi, Andrea Samoré. Deep-plug-and-play proximal Gauss-Newton method with applications to nonlinear, ill-posed inverse problems. Inverse Problems and Imaging, 2023, 17(6): 1226-1248. doi: 10.3934/ipi.2023014
* Francesco Colibazzi, Damiana Lazzaro, Serena Morigi, Andrea Samoré. Learning Nonlinear Electrical Impedance Tomography. J Sci Comput, 2022, 90(58). https://doi.org/10.1007/s10915-021-01716-4
