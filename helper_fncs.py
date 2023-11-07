import scipy as sp
import KTCMeshing
import numpy as np
from scipy.spatial import Delaunay
import torch
import KTCRegularization
import KTCFwd
from torch.nn.functional import interpolate
#from KTCAux_luca import simulateConductivityNew
from torchvision.transforms.functional import vflip
import deepinv as dinv



def get_mask(difficulty):
    Injref = sp.io.loadmat('TrainingData/ref.mat')["Injref"]  # load the reference data
    vincl = np.ones(((32 - 1), 76), dtype=bool)  # which measurements to include in the inrversion
    # adjust operators for different levels of difficulties
    rmind = np.arange(0, 2 * (difficulty - 1), 1)  # electrodes whose data is removed
    # adjust matrix according to the difficulty level
    for ii in range(0, 75):
        for jj in rmind:
            if Injref[jj, ii]:
                vincl[:, ii] = 0
            vincl[jj, :] = 0

    return np.array(vincl, bool).flatten()

class EIT_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, difficulty=1, train=True):
        super().__init__()
        self.data = dinv.datasets.HDF5Dataset(path=path, train=train)
        self.train = train

        self.mask = get_mask(difficulty)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y[self.mask, :]


class Network(torch.nn.Module):
    def __init__(self, backbone, regularization, difficulty, device, pixels, train_first=False):
        super(Network, self).__init__()
        self.backbone = backbone
        self.W = LinearInverse(difficulty, regularization, device, train=train_first)
        self.Interpolator = MeshInterpolator(load_mesh('Mesh_sparse.mat', index=1), pixels, device)
        self.pixels = pixels

    def forward(self, x, physics=None, **kwargs):
        x = self.W(x)
        x = self.Interpolator(x)
        x = self.backbone(x)
        return x


def load_mesh(file, index=1):
    g_str = 'g' if index == 1 else 'g2'
    h_str = 'H' if index == 1 else 'H2'
    elfaces_str = 'elfaces' if index == 1 else 'elfaces2'
    node_str = 'Node' if index == 1 else 'Node2'
    element_str = 'Element' if index == 1 else 'Element2'
    elementE_str = 'ElementE' if index == 1 else 'Element2E'

    # load premade finite element mesh (made using Gmsh, exported to Matlab and saved into a .mat file)
    mat_dict_mesh = sp.io.loadmat(file)  # Mesh_dense
    g = mat_dict_mesh[g_str]  # node coordinates
    H = mat_dict_mesh[h_str]  # indices of nodes making up the triangular elements
    elfaces = mat_dict_mesh[elfaces_str][0].tolist()  # indices of nodes making up the boundary electrodes

    # Element structure
    ElementT = mat_dict_mesh[element_str]['Topology'].tolist()
    for k in range(len(ElementT)):
        ElementT[k] = ElementT[k][0].flatten()
    ElementE = mat_dict_mesh[elementE_str].tolist()  # marks elements which are next to boundary electrodes
    for k in range(len(ElementE)):
        if len(ElementE[k][0]) > 0:
            ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
        else:
            ElementE[k] = []

    # Node structure
    NodeC = mat_dict_mesh[node_str]['Coordinate']
    NodeE = mat_dict_mesh[node_str]['ElementConnection']  # marks which elements a node belongs to
    nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
    for k in range(NodeC.shape[0]):
        nodes[k].ElementConnection = NodeE[k][0].flatten()
    elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
    for k in range(len(ElementT)):
        elements[k].Electrode = ElementE[k]

    return KTCMeshing.Mesh(H, g, elfaces, nodes, elements)



class LinearInverse(torch.nn.Module):
    def __init__(self, difficulty, regularization, device, train=False):
        super(LinearInverse, self).__init__()
        Nel = 32
        # load injection pattern from ref file
        inputFolder = 'TrainingData'
        mat_dict = sp.io.loadmat(inputFolder + '/ref.mat')  # load the reference data
        Injref = mat_dict["Injref"]  # current injections
        Mpat = mat_dict["Mpat"]  # current injections
        Uelref = mat_dict["Uelref"]  # current injections

        vincl = get_mask(difficulty)

        Mesh = load_mesh('Mesh_sparse.mat', index=1)
        Mesh2 = load_mesh('Mesh_sparse.mat', index=2)
        solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)
        # measurement pattern
        z = 1e-6*np.ones((Nel, 1))  # contact impedances

        sigma0 = np.ones((len(Mesh.g), 1))  # linearization point
        corrlength = 1 * 0.115  # used in the prior
        var_sigma = 0.05 ** 2  # 0.05 ** 2 #prior variance
        mean_sigma = sigma0
        smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)

        # set up the noise model for inversion
        noise_std1 = 0.05  # standard deviation for first noise component (relative to each voltage measurement)
        noise_std2 = 0.01  # standard deviation for second noise component (relative to the largest voltage measurement)
        solver.SetInvGamma(noise_std1, noise_std2, Uelref)

        # linearized reconstruction
        J = solver.Jacobian(sigma0, z)
        mask = np.array(vincl, bool)

        recon_matrices = []
        for reg in regularization:
            reco_matrix = np.linalg.inv(J.T @ solver.InvGamma_n[np.ix_(mask, mask)] @ J + reg * smprior.L.T @ smprior.L)
            reco_matrix = reco_matrix @ J.T @ solver.InvGamma_n[np.ix_(mask, mask)]
            recon_matrices.append(reco_matrix)

        H, _ = recon_matrices[0].shape

        W = torch.zeros((len(recon_matrices), H, int(np.sum(vincl))), dtype=torch.float32, device=device)

        for i in range(len(recon_matrices)):
            W[i, :, :] = torch.Tensor(recon_matrices[i]).to(device)

        self.W = W
        if train:
            self.W = torch.nn.Parameter(self.W, requires_grad=True)

    def forward(self, x):
        return self.W @ x.unsqueeze(1)


class MeshInterpolator(torch.nn.Module):
    def __init__(self, Mesh, pixels, device):
        super(MeshInterpolator, self).__init__()
        self.pixels = pixels

        pixwidth = 0.23 / pixels
        pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, pixels)
        pixcenter_y = pixcenter_x
        X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
        pixcenters = np.column_stack((X.ravel(), Y.ravel()))

        # build INTMAP
        g = Mesh.g
        pts = pixcenters

        TR = Delaunay(g)
        Hdel = TR.simplices
        invX = [np.linalg.inv(np.column_stack((g[pp, :], np.ones(3)))) for pp in Hdel]
        np_pts = len(pts)
        Ic = np.zeros((np_pts, 3))
        Iv = np.zeros((np_pts, 3))
        Element = TR.find_simplex(pts)
        nans = np.zeros(np_pts)
        for k in range(np_pts):
            tin = Element[k]
            Phi = np.zeros(3)
            if not np.isnan(tin):
                if tin >= 0:
                    iXt = invX[tin]
                    for gin in range(3):
                        Phi[gin] = np.dot(np.append(pts[k, :], 1), iXt[:, gin])
                    Ic[k, :] = Hdel[tin, :]
                else:
                    Ic[k, :] = 1
                    nans[k] = 1
                    Iv[k, :] = 1
                Iv[k, :] = Phi
            else:
                Ic[k, :] = 1
                nans[k] = 1
                Iv[k, :] = 1

        INTPMAT = np.zeros((np_pts, len(Mesh.g)))
        for row in range(np_pts):
            INTPMAT[row, Ic[row].astype(int)] = Iv[row]
        INTPMAT[nans == 1, :] = 0

        self.W = torch.Tensor(INTPMAT).to(device)


    def forward(self, x):
        x = self.W @ x
        x = x.reshape(x.shape[0], x.shape[1], self.pixels, self.pixels)
        if self.pixels != 256:
            x = interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = vflip(x)
        return x

def generate_sample(solver, mesh, z, Iel_ref_sim, interpolator):
    # simulated initial conductivity and the change + segment simulated images
    n_incl = np.random.randint(1, 3)
    sigma, delta_sigma, _ = simulateConductivityNew(mesh, n_incl)

    # simulate data
    Iel_true = solver.SolveForward(sigma + delta_sigma, z)

    # add some noise
    Iel2_noisy = Iel_ref_sim + solver.InvLn * np.random.randn(Iel_ref_sim.shape[0], 1)
    Iel_noisy = Iel_true + solver.InvLn * np.random.randn(Iel_true.shape[0], 1)
    deltaI = Iel_noisy - Iel2_noisy

    #  GT images on sparse mesh and assign class there using levels computed before
    delta_pixgrid_fine = interpolator(torch.Tensor(delta_sigma).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).numpy()
    delta_pixgrid_out = np.zeros((3, 256, 256))
    delta_pixgrid_out[0, delta_pixgrid_fine == 0] = 1
    delta_pixgrid_out[1, delta_pixgrid_fine < 0] = 1
    delta_pixgrid_out[2, delta_pixgrid_fine > 0] = 1

    return delta_pixgrid_out, deltaI