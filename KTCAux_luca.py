import numpy as np
import scipy as sp
from scipy.spatial import Delaunay, ConvexHull
from matplotlib.path import Path

def Interpolate2Newmesh2DNode(g, H, Node, f, pts, INTPMAT):

    if INTPMAT == []:
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
        INTPMAT = np.zeros((np_pts, f.size))
        for row in range(np_pts):
            INTPMAT[row, Ic[row].astype(int)] = Iv[row]
        INTPMAT[nans == 1, :] = 0

    f_newgrid = np.dot(INTPMAT, f)
    return f_newgrid, INTPMAT, Element

def setMeasurementPattern(Nel):
    Inj = np.matrix(np.eye(Nel))
    gnd = np.matrix(np.eye(Nel,k=-1))
    Mpat = np.matrix(Inj[:, :Nel-1] - gnd[:, :Nel-1])
    vincl = np.ones(Nel * (Nel - 1), dtype=bool)
    return Inj, Mpat, vincl

def simulateConductivity(Meshsim, inclusiontypes):
    sigma = np.ones((Meshsim.g.shape[0], 1))
    contrast = -0.5
    cp = np.array([0.5*0.115, 0.5*0.115])
    r = 0.2*0.115
    ind = np.where(np.linalg.norm(Meshsim.g.T - cp[:, None], axis=0) <= r)[0]
    delta_sigma = np.zeros_like(sigma)
    delta_sigma[ind] = contrast
    cp = np.array([0, 0])
    r = 0.2*0.115
    ind = np.where(np.linalg.norm(Meshsim.g.T - cp[:, None], axis=0) <= r)[0]
    if inclusiontypes == 2:
        delta_sigma[ind] = contrast
    else:
        delta_sigma[ind] = abs(contrast)
    sigma2 = sigma + delta_sigma
    return sigma, delta_sigma, sigma2

def simulateConductivityNew(Meshsim, n_incl):
    
    # we do not want the inclusion to be closer than d0 to the boundary
    R = 0.115 # the radius of the original domain
    d0 = 0.2*R # minimum tolerated distance from the boundary
    indices_d0 = np.where( np.square(Meshsim.g.T[0,:])+np.square(Meshsim.g.T[1,:]) >= (R-d0)**2)
    ind_d0 = np.zeros((Meshsim.g.T.shape[1]))
    ind_d0[indices_d0] = 1

    # generate shapes until they do not overlap and are not too close to the boundary
    #single_incl_shape = 6*np.ones(n_incl) 
    #single_incl_holes = 0*np.ones(n_incl)
    single_incl_shape = np.random.randint(6,size=n_incl) # random array of shapes
    single_incl_holes = np.random.randint(2,size=n_incl) # for each shape, assign a boolean variable determining whether it has holes
    bad_inclusion = True
    attempts = 0
    while bad_inclusion:
        ind_fun = np.zeros((Meshsim.g.T.shape[1], n_incl))
        for i in range(n_incl):
            ind_fun[:,i] = createSingleInclusion(Meshsim, single_incl_shape[i], single_incl_holes[i], n_incl)
        check_overlap = np.sum(ind_fun,axis=1) + ind_d0
        attempts = attempts+1
        if np.mod(attempts,1000)==0:
           print(attempts)
        if (np.max(check_overlap)<=1) & (np.min(np.max(ind_fun,axis=0))>0):
            bad_inclusion = False
    print('Admissible shape generated in', attempts, 'attempts\n')  

    contrast = np.multiply(np.random.choice([-1.,1.],n_incl), 0.1 + 0.4*np.random.random((1,n_incl))) # uniform random in (-0.5, -0.1) U (0.1, 0.5)
    ref_value = 1 # 0.7 + 0.6*np.random.random()    # uniform random in (0.7, 1.3)    
    sigma = ref_value*np.ones((Meshsim.g.shape[0], 1))
    delta_sigma = np.sum(np.multiply(ind_fun,contrast),1).reshape(-1,1)
    
    sigma2 = sigma + delta_sigma
    
    return sigma, delta_sigma, sigma2

def createSingleInclusion(Meshsim, incl_shape, ind_holes,n_incl):
    ind_fun = np.zeros((Meshsim.g.shape[0]))
    R = 0.115 # the radius of the original domain
    d0 = 0.2*R
    match incl_shape:
        case 0: # circles
            # cx = 0
            # cy = 0
            # rad = 0.3*R
            # rad2 = 0.1*R
            cx = -R + 2*R*np.random.random() # uniform random in (-R, R)
            cy = -R + 2*R*np.random.random() # uniform random in (-R, R)
            rad = 0.2*R + 0.3*R*np.random.random()   # uniform random in (0.3R, 0.5R)
            rad2 = 0.05*R + (rad-0.1*R)*np.random.random()  # uniform random in (0.1R, 0.5R)
            if ind_holes==0:
               indices = np.where( np.square(Meshsim.g.T[0,:]-cx)+np.square(Meshsim.g.T[1,:]-cy) <= rad**2)  
            else:          
               indices = np.where( np.logical_and( np.square(Meshsim.g.T[0,:]-cx)+ np.square(Meshsim.g.T[1,:]-cy) <= rad**2, \
                                 np.square(Meshsim.g.T[0,:]-cx)+ np.square(Meshsim.g.T[1,:]-cy) >= rad2**2) )  

        case 1: # squares
            # cx = 0
            # cy = 0
            # side = 0.4*R
            # angle = np.pi/3
            cx = -R + 2*R*np.random.random() # uniform random in (-R, R)
            cy = -R + 2*R*np.random.random() # uniform random in (-R, R)
            side = 0.2*R + 0.3*R*np.random.random()    # uniform random in (0.2*R, 0.5*R)
            fact = 0.2 + 0.5*np.random.rand()  # uniform random in [0,1) for scaling sides (same)
            angle = np.pi/2*np.random.random() # uniform random in (0,pi/2)
            x_shift_rot = np.cos(angle)*(Meshsim.g.T[0,:]-cx) + np.sin(angle)*(Meshsim.g.T[1,:]-cy)           
            y_shift_rot = -np.sin(angle)*(Meshsim.g.T[0,:]-cx) + np.cos(angle)*(Meshsim.g.T[1,:]-cy)
            if ind_holes==0:
               indices = np.where( np.logical_and(np.abs(x_shift_rot) <= side/2, np.abs(y_shift_rot) <= side/2))
            else: 
               indices = np.where( np.logical_and(np.logical_and(np.abs(x_shift_rot) <= side/2, np.abs(y_shift_rot) <= side/2 ), \
                                                np.logical_or(np.abs(x_shift_rot) > fact*side/2, np.abs(y_shift_rot) > fact*side/2))) 
               

        case 2: # rectangles
            #  cx = 0
            #  cy = 0
            #  side1 = 0.5*R
            #  side2 = 0.3*R
            #  angle = np.pi/3            
             cx = -R + 2*R*np.random.random() # uniform random in (-R, R)
             cy = -R + 2*R*np.random.random() # uniform random in (-R, R)
             side1 = 0.2*R + 0.3*R*np.random.random()    # uniform random in (0.2*R, 0.5*R)
             side2 = 0.1*R + 0.2*R*np.random.random()    # uniform random in (0.1*R, 0.3*R)
             fact = 0.2 + 0.5*np.random.rand()  # uniform random in [0,1) for scaling sides (same)
             angle = np.pi/2*np.random.random() # uniform random in (0,pi/2)
             x_shift_rot = np.cos(angle)*(Meshsim.g.T[0,:]-cx) + np.sin(angle)*(Meshsim.g.T[1,:]-cy)           
             y_shift_rot = -np.sin(angle)*(Meshsim.g.T[0,:]-cx) + np.cos(angle)*(Meshsim.g.T[1,:]-cy)
             if ind_holes==0:
                 indices = np.where( np.logical_and( np.abs(x_shift_rot) <= side1/2, np.abs(y_shift_rot) <= side2/2))
             else:
                 indices = np.where( np.logical_and( np.logical_and(np.abs(x_shift_rot) <= side1 /2, np.abs(y_shift_rot) <= side2 /2), \
                                                     np.logical_or(np.abs(x_shift_rot) > fact*side1 /2, np.abs(y_shift_rot) >  fact*side2 /2) ))

        case 3: # star
            # cx = 0
            # cy = 0
            # angle = np.pi/3
            # N_tips = 4
            # r_ex = 0.5*R
            # r_in = 0.2*R
            cx = -R + 2*R*np.random.random() # uniform random in (-R, R)
            cy = -R + 2*R*np.random.random() # uniform random in (-R, R)
            N_tips = np.random.randint(3,6) # random number among 3, 4, or 5
            angle = 2*np.pi/N_tips*np.random.random() # uniform random in (0, 2*pi/N_tips)
            r_ex = 0.3*R + 0.2*R*np.random.random() # uniform random in (0.3*R, 0.5*R)
            r_in = r_ex/2
            fact = 0.2 + 0.6*np.random.rand()  # rescaling factor
            Rho = np.sqrt( np.square(Meshsim.g.T[0,:]-cx) + np.square(Meshsim.g.T[1,:]-cy))
            Theta = np.arctan2(Meshsim.g.T[1,:]-cy, Meshsim.g.T[0,:]-cx) - angle
            if ind_holes==0:
                indices = np.where(Rho<((r_ex+r_in)/2 + (r_ex-r_in)/2*np.cos(N_tips*Theta)))
            else:
                indices = np.where(np.logical_and( Rho<((r_ex+r_in)/2 + (r_ex-r_in)/2*np.cos(N_tips*Theta)), \
                                   Rho>=((r_ex+r_in)*fact/2 + (r_ex-r_in)*fact/2*np.cos(N_tips*Theta))))

        case 4: # horseshoe
            # cx = 0
            # cy = 0
            # side = 0.35*R
            # angle = np.pi/6
            cx = -R + 2*R*np.random.random() # uniform random in (-R, R)
            cy = -R + 2*R*np.random.random() # uniform random in (-R, R)
            side = 0.3*R + 0.2*R*np.random.random()    # uniform random in (0.3*R, 0.5*R)
            angle = 2*np.pi*np.random.random() # uniform random in (0, 2*pi)
            x_shift_rot = np.cos(angle)*(Meshsim.g.T[0,:]-cx) + np.sin(angle)*(Meshsim.g.T[1,:]-cy)           
            y_shift_rot = -np.sin(angle)*(Meshsim.g.T[0,:]-cx) + np.cos(angle)*(Meshsim.g.T[1,:]-cy)
            indices = np.where( np.logical_and(np.logical_and( np.abs(x_shift_rot) <= side/2, np.abs(y_shift_rot) <= side/2) , \
                                               np.logical_or( np.abs(x_shift_rot) > side/4, y_shift_rot > side/4)))
            
        case 5: # triangle

            bad_triang = True
            scaling = 40

            if n_incl == 3:
                bound_up = 4*np.pi*R**2/scaling
            else:
               bound_up = np.Inf
            
            while bad_triang:
               cx = -R + 2*R*np.random.random(3)  # fix random x coordinates 
               cy = -R + 2*R*np.random.random(3)  # fix random y coordinates 
               # check if all different (how to draw without replacement?)
               area_triang =sp.spatial.ConvexHull(np.array([[cx[0], cy[0]], [cx[1], cy[1]], [cx[2], cy[2]]]))
               if (area_triang.volume >= np.pi*R**2/scaling) & (area_triang.volume < bound_up) & (area_triang.area < 8*np.sqrt(area_triang.volume)):
                  bad_triang = False
               # is rotation needed since we generate random triangles?
               # angle = np.pi/6 
               # x_shift_rot = np.cos(angle)*(Meshsim.g.T[0,:]-cx) + np.sin(angle)*(Meshsim.g.T[1,:]-cy)           
               # y_shift_rot = -np.sin(angle)*(Meshsim.g.T[0,:]-cx) + np.cos(angle)*(Meshsim.g.T[1,:]-cy)
               x_shift_rot = Meshsim.g.T[0,:]
               y_shift_rot = Meshsim.g.T[1,:]

            # indices = np.where(np.logical_or(np.logical_and(np.logical_and( (cx[1]-cx[0])*(y_shift_rot-cy[0]) - (cy[1]-cy[0])*(x_shift_rot-cx[0]) < 0, \
            #                                                                 (cx[2]-cx[1])*(y_shift_rot-cy[1]) - (cy[2]-cy[1])*(x_shift_rot-cx[1]) < 0 ), \
            #                                                                  (cx[0]-cx[2])*(y_shift_rot-cy[2]) - (cy[0]-cy[2])*(x_shift_rot-cx[2]) < 0 ), \
            #                                                                  np.logical_and(np.logical_and( (cx[1]-cx[0])*(y_shift_rot-cy[0]) - (cy[1]-cy[0])*(x_shift_rot-cx[0]) > 0, \
            #                                                                  (cx[2]-cx[1])*(y_shift_rot-cy[1]) - (cy[2]-cy[1])*(x_shift_rot-cx[1]) > 0 ), \
            #                                                                  (cx[0]-cx[2])*(y_shift_rot-cy[2]) - (cy[0]-cy[2])*(x_shift_rot-cx[2]) > 0 )))
            points = np.array([[cx[0], cy[0]], [cx[1], cy[1]], [cx[2], cy[2]]])
            hull = ConvexHull(points)
            hull_path = Path(points[hull.vertices])

            indices = np.where(hull_path.contains_points(np.transpose([x_shift_rot,y_shift_rot])))

        case 6: #general quadrilater

            bad_quadr = True
            scaling = 40

            if n_incl == 3:
                bound_up = 4*np.pi*R**2/scaling
            else:
               bound_up = np.Inf

            while bad_quadr:
                cx = -R + 2*R*np.random.random(4)  # fix random x coordinates 
                cy = -R + 2*R*np.random.random(4)  # fix random y coordinates 
                area_quadr =sp.spatial.ConvexHull(np.array([[cx[0], cy[0]], [cx[1], cy[1]], [cx[2], cy[2]], [cx[3], cy[3]]]))
                if (area_quadr.volume >= np.pi*R**2/scaling) &  (area_quadr.volume < bound_up) & (area_quadr.area < 10*np.sqrt(area_quadr.volume)):
                  bad_quadr = False
           
            # is rotation needed since we generate random triangles?
            # angle = np.pi/6 
            # x_shift_rot = np.cos(angle)*(Meshsim.g.T[0,:]-cx) + np.sin(angle)*(Meshsim.g.T[1,:]-cy)           
            # y_shift_rot = -np.sin(angle)*(Meshsim.g.T[0,:]-cx) + np.cos(angle)*(Meshsim.g.T[1,:]-cy)
            x_shift_rot = Meshsim.g.T[0,:]
            y_shift_rot = Meshsim.g.T[1,:]

            points = np.array([[cx[0], cy[0]], [cx[1], cy[1]], [cx[2], cy[2]], [cx[3], cy[3]]])
            hull = ConvexHull(points)
            hull_path = Path(points[hull.vertices])

            indices = np.where(hull_path.contains_points(np.transpose([x_shift_rot,y_shift_rot])))

            # indices = np.where(np.logical_or(np.logical_and(np.logical_and(np.logical_and( (cx[1]-cx[0])*(y_shift_rot-cy[0]) - (cy[1]-cy[0])*(x_shift_rot-cx[0]) < 0, \
            #                                                                  (cx[2]-cx[1])*(y_shift_rot-cy[1]) - (cy[2]-cy[1])*(x_shift_rot-cx[1]) < 0 ), \
            #                                                                  (cx[3]-cx[2])*(y_shift_rot-cy[2]) - (cy[3]-cy[2])*(x_shift_rot-cx[2]) < 0 ), \
            #                                                                  (cx[0]-cx[3])*(y_shift_rot-cy[3]) - (cy[0]-cy[3])*(x_shift_rot-cx[3]) < 0 ), \
            #                                                                  np.logical_and(np.logical_and(np.logical_and( (cx[1]-cx[0])*(y_shift_rot-cy[0]) - (cy[1]-cy[0])*(x_shift_rot-cx[0]) > 0, \
            #                                                                  (cx[2]-cx[1])*(y_shift_rot-cy[1]) - (cy[2]-cy[1])*(x_shift_rot-cx[1]) > 0 ), \
            #                                                                  (cx[3]-cx[2])*(y_shift_rot-cy[2]) - (cy[3]-cy[2])*(x_shift_rot-cx[2]) > 0 ), \
            #                                                                  (cx[0]-cx[3])*(y_shift_rot-cy[3]) - (cy[0]-cy[3])*(x_shift_rot-cx[3]) > 0 )))

          
            

    
    ind_fun[indices] = 1.
    return ind_fun


def interpolateRecoToPixGrid(deltareco, Mesh):
    pixwidth = 0.23 / 256
    # pixcenter_x = np.arange(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, pixwidth)
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
    pixcenters = np.column_stack((X.ravel(), Y.ravel()))
    deltareco_pixgrid = Interpolate2Newmesh2DNode(Mesh.g, Mesh.H, Mesh.Node, deltareco, pixcenters, [])
    deltareco_pixgrid = deltareco_pixgrid[0]
    deltareco_pixgrid = np.flipud(deltareco_pixgrid.reshape(256, 256))
    return deltareco_pixgrid

