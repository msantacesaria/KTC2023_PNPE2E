import deepinv as dinv
import torch
import numpy as np
import scipy as sp
from helper_fncs import Network, EIT_Dataset, get_mask
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import KTCScoring
import KTCFwd
import KTCMeshing
import KTCRegularization
import KTCPlotting
import KTCAux
import scipy.io as spio
from torch_geometric.data import Data
from torch_geometric.nn import GraphUNet
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.random.seed(77) #77
MODEL_PTH = './weights_denoiser.pth'
#inputFolderMask='Mask_Synthetic/'
# device = torch.device('cpu')
# device = dinv.utils.get_freer_gpu()
device = torch.device('cpu')


class GUNet(torch.nn.Module):
    def __init__(self, data):
        super(GUNet, self).__init__()
        self.unet = GraphUNet(data.num_node_features, 64, data.num_node_features, depth=5, pool_ratios=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.unet(x, edge_index)
        return x

def pnp_net(Uel,edge_index,LTL,solver,denoiser,z,sigma,Mesh,its_max_PGN,rel_ch_PGN,alfa,InvG,step_size):

  #PGN-TV
     
    rel_ch = 1
    it = 0
    sigma0=sigma.copy()
    while (it < its_max_PGN) and (rel_ch > rel_ch_PGN):
        #print('OUTER Iteration:{}'.format(it))
        sigma_old=sigma.copy()

        # forward step
        out_fw_step,matrixreg = forward_step(Uel, LTL, solver,sigma_old, z, sigma0, alfa,InvG,step_size)

        '''
        # plot forward
        sgplot = KTCPlotting.SigmaPlotter(Meshsim, [2, 3], 'jet')
        title="fwd"+str(it)
        sgplot.basicplot(out_fw_step,title)
        '''
        # backward step: deep denoiser
        sigma_net=backward_step_net(out_fw_step,denoiser,edge_index)
        sigma_net=sigma_net.to(torch.device('cpu'))
        sigma=sigma_net.detach().numpy()

        '''
        # plot backward
        sgplot = KTCPlotting.SigmaPlotter(Meshsim, [2, 3], 'jet')
        title="bkd"+str(it)
        sgplot.basicplot(sigma,title)
        '''
        
        it += 1
        rel_ch = np.linalg.norm(sigma - sigma_old) / np.linalg.norm(sigma)
        print('it {} relative error: {}'.format(it,rel_ch))

    return sigma


def backward_step_net(z,denoiser,edge_index):
    z = z.detach().numpy()
    x_in =torch.tensor(z, dtype=torch.float32)
    data = Data(x=x_in, edge_index=edge_index).to(device)

    outinf = denoiser(data)
    return outinf


def forward_step(Uel,LTL_in, solver, sigma,z,sigma0,alfa,InvG_in,step_size):
    LTL = torch.from_numpy(LTL_in).to(torch.float32)
    InvG = torch.from_numpy(InvG_in).to(torch.float32)

    sigma0T=torch.from_numpy(sigma0).to(torch.float32)
    sigmaT = torch.from_numpy(sigma).to(torch.float32)
    Usim = solver.SolveForward(sigma, z) #forward solution at the previous iteration

    res =  torch.from_numpy(Usim).to(torch.float32) - Uel
    J = solver.Jacobian(sigma, z)
    jm = torch.from_numpy(J).to(torch.float32)
    JTJ= torch.mm(torch.transpose(jm,0,1), torch.mm(torch.diag(InvG),jm))

    # nn=JTJ.shape[0]
    # LTL = torch.eye(nn)
    matrixreg = JTJ + alfa*LTL

    tn = torch.mm(torch.transpose(jm, 0, 1), torch.mul(torch.unsqueeze(InvG,1),res)) + alfa*torch.mm(LTL,sigmaT - sigma0T)
    agg = torch.linalg.solve(matrixreg,tn.to(torch.float32))
    zz = sigmaT - step_size*agg

    return zz,matrixreg


# model
regularization = [.001]
pixels = 64

aver = False

def main(inputFolder,outputFolder,difficulty):
    
    # from E2E
    Uelref = sp.io.loadmat(inputFolder + '/ref.mat')["Uelref"] #measured voltages from water chamber
    mask = get_mask(difficulty)
    backbone = dinv.models.UNet(in_channels=len(regularization), out_channels=3, scales=4).to(device)
    model = Network(backbone, regularization, difficulty=difficulty, device=device, pixels=pixels, train_first=True)
    p = Path(f"models/difficulty_{difficulty}_learnedlinear.pth.tar")
    model.load_state_dict(torch.load(p)['state_dict'])
    # model.load_state_dict(torch.load(p,map_location=torch.device('cpu'))['state_dict'])   # for CPU
    model.eval()

    # from PnP
    Nel = 32  # number of electrodes
    ### load premade finite element mesh (made using Gmsh, exported to Matlab and saved into a .mat file)
    mat_dict_mesh = spio.loadmat('Mesh_sparse.mat') #Mesh_dense
    g = mat_dict_mesh['g'] #node coordinates
    H = mat_dict_mesh['H'] #indices of nodes making up the triangular elements
    elfaces = mat_dict_mesh['elfaces'][0].tolist() #indices of nodes making up the boundary electrodes
    ### Element structure
    ElementT = mat_dict_mesh['Element']['Topology'].tolist()
    for k in range(len(ElementT)):
        ElementT[k] = ElementT[k][0].flatten()
    ElementE = mat_dict_mesh['ElementE'].tolist() #marks elements which are next to boundary electrodes
    for k in range(len(ElementE)):
        if len(ElementE[k][0]) > 0:
            ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
        else:
            ElementE[k] = []
    ### Node structure
    NodeC = mat_dict_mesh['Node']['Coordinate']
    NodeE = mat_dict_mesh['Node']['ElementConnection'] #marks which elements a node belongs to
    nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
    for k in range(NodeC.shape[0]):
        nodes[k].ElementConnection = NodeE[k][0].flatten()
    elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
    for k in range(len(ElementT)):
        elements[k].Electrode = ElementE[k]
    ### 2nd order mesh data
    H2 = mat_dict_mesh['H2']
    g2 = mat_dict_mesh['g2']
    elfaces2 = mat_dict_mesh['elfaces2'][0].tolist()
    ElementT2 = mat_dict_mesh['Element2']['Topology']
    ElementT2 = ElementT2.tolist()
    for k in range(len(ElementT2)):
        ElementT2[k] = ElementT2[k][0].flatten()
    ElementE2 = mat_dict_mesh['Element2E']
    ElementE2 = ElementE2.tolist()
    for k in range(len(ElementE2)):
        if len(ElementE2[k][0]) > 0:
            ElementE2[k] = [ElementE2[k][0][0][0], ElementE2[k][0][0][1:len(ElementE2[k][0][0])]]
        else:
            ElementE2[k] = []
    NodeC2 = mat_dict_mesh['Node2']['Coordinate']  # ok
    NodeE2 = mat_dict_mesh['Node2']['ElementConnection']  # ok
    nodes2 = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC2]
    for k in range(NodeC2.shape[0]):
        nodes2[k].ElementConnection = NodeE2[k][0].flatten()
    elements2 = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT2]
    for k in range(len(ElementT2)):
        elements2[k].Electrode = ElementE2[k]
    Meshsim = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
    Mesh2sim = KTCMeshing.Mesh(H2,g2,elfaces2,nodes2,elements2)
    Mesh = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
    ### set up forward solver
    nTri=H.shape[0]
    nv=g.shape[0]
    Ad= np.zeros((nv, nv), dtype=int)
    for i in range(nTri):
            Ad[H[i,0],H[i,1]]=1
            Ad[H[i,1],H[i,0]]=1
            Ad[H[i,0],H[i,2]]=1
            Ad[H[i,2],H[i,0]]=1
            Ad[H[i,1],H[i,2]]=1
            Ad[H[i,2],H[i,1]]=1
    Ad=torch.tensor(Ad)
    edge_index = Ad.nonzero().t().contiguous()
    z = 1e-6*np.ones((Nel, 1))  # contact impedances
    
    
    mat_dict = spio.loadmat(inputFolder + '/ref.mat') #load the reference data
    Injref = mat_dict["Injref"] #current injections
    Uelref = mat_dict["Uelref"]
    Mpat = mat_dict["Mpat"] #voltage measurement pattern
    vincl = np.ones(((Nel - 1),76), dtype=bool) #which measurements to include in the inversion
    rmind = np.arange(0,2 * (difficulty - 1),1) #electrodes whose data is removed
    
        #remove measurements according to the difficulty level
    for ii in range(0,75):
            for jj in rmind:
                if Injref[jj,ii]:
                    vincl[:,ii] = 0
                vincl[jj,:] = 0
    solver = KTCFwd.EITFEM(Mesh2sim, Injref, Mpat, vincl)
    
    vincl = vincl.T.flatten()
    
    noise_std1 = 0.05;  # standard deviation for first noise component (relative to each voltage measurement)
    noise_std2 = 0.01;  # standard deviation for second noise component (relative to the largest voltage measurement)
    solver.SetInvGamma(noise_std1, noise_std2, Uelref)
    #InvG = solver.InvGamma_n#.diagonal()
    
    
    sigma0 = np.ones((len(Mesh.g), 1))
    corrlength = 1 * 0.115 #used in the prior
    var_sigma = 0.05 ** 2 #prior variance
    mean_sigma = sigma0
    smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)
    LTL= smprior.L.T @ smprior.L  #Regularizer
    
    data = Data(x=torch.tensor(sigma0, dtype=torch.float32), edge_index=edge_index)
    denoiser = GUNet(data).to(device)
    denoiser.load_state_dict(torch.load(MODEL_PTH, map_location=torch.device('cpu')), strict=False)
    
    # PnP Parameters
    its_max_PGN = 10    #Maximum number of OUTER iterations for Proximal Gradient Newton- TV (PGN_TV)
    rel_ch_PGN = 5e-4     #relative change
    alfa = 0.05   #alfa
    step_size = 5e-2

    # Get a list of .mat files in the input folder
    mat_files = glob.glob(inputFolder + '/data*.mat')
    for objectno in  range (0,len(mat_files)): #compute the reconstruction for each input file
        Uel = sp.io.loadmat(mat_files[objectno])["Uel"]
        deltaU = Uel - Uelref
        y = np.zeros((1, np.sum(mask), 1))

        y[0,:,:]=deltaU[mask,:]

        y = torch.Tensor(y).to(device)
        x_net = model(y)
        x_net = torch.argmax(x_net, dim=1, keepdim=True)
        
        # interpolate the reconstruction into a pixel image
        e2e_reco = np.reshape(x_net.detach().cpu().numpy(),[256,256])
        
        
        mat_dict2 = spio.loadmat(mat_files[objectno])
        Inj = mat_dict2["Inj"]
        Uel = mat_dict2["Uel"]
        U0 = solver.SolveForward(sigma0, z)
        vm_32 = torch.from_numpy(Uel[vincl]-Uelref[vincl]+U0).to(torch.float32)
        mask = np.array(vincl, bool)
        InvG=solver.InvGamma_n[np.ix_(mask,mask)].diagonal()
        reco = pnp_net(vm_32,edge_index,LTL,solver,denoiser,z,sigma0,Mesh,its_max_PGN,rel_ch_PGN,alfa,InvG,step_size)
        reco_pixgrid = KTCAux.interpolateRecoToPixGrid(reco, Mesh)
        
        #threshold the image histogram using Otsu's method
        level, x = KTCScoring.Otsu2(reco_pixgrid.flatten(), 256, 7)
        reco_pixgrid_segmented = np.zeros_like(reco_pixgrid)
        ind0 = reco_pixgrid < x[level[0]]
        ind1 = np.logical_and(reco_pixgrid >= x[level[0]],reco_pixgrid <= x[level[1]])
        ind2 = reco_pixgrid > x[level[1]]
        inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
        bgclass = inds.index(max(inds)) #background class
            
        if bgclass==0:
          reco_pixgrid_segmented[ind1] = 2
          reco_pixgrid_segmented[ind2] = 2
        if bgclass==1:
          reco_pixgrid_segmented[ind0] = 1
          reco_pixgrid_segmented[ind2] = 2
        if bgclass==2:
          reco_pixgrid_segmented[ind0] = 1
          reco_pixgrid_segmented[ind1] = 1
                     
        reconstruction = reco_pixgrid_segmented
        for i in range(256):
          for j in range(256):
           if e2e_reco[i,j]==0:
             reconstruction[i,j]=0

        mdic = {"reconstruction": reconstruction}
        print(outputFolder + '/' + str(objectno + 1) + '.mat')
        sp.io.savemat( outputFolder + '/' + str(objectno + 1) + '.mat',mdic)