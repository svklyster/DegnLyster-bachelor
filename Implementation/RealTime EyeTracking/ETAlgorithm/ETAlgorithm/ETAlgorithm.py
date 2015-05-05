import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


# ransac funtion, go!

def convert_conic_parameters_to_ellipse_parameters(c):
    theta = np.arctan2(c[1], c[0]-c[2])/2
    ct = np.cos(theta)
    st = np.sin(theta)
    ap = c[0]*ct*ct+c[1]*ct*st+c[2]*st*st
    cp = c[0]*st*st-c[1]*ct*st+c[2]*ct*ct

    T = np.array([np.transpose([c[0], c[1]/2]),np.transpose([c[1]/2, c[2]])])
    t = np.linalg.solve(-(2*T),np.transpose(np.array([c[3],c[4]])))
    cx = t[0]
    cy = t[1]

    val = np.dot(np.dot(np.transpose(t),T),t)
    scale_inv = val-c[5]

    a = np.sqrt(scale_inv/(ap+0.J))
    #bs = np.array([scale_inv/(cp+0.J)])
    b = np.sqrt(scale_inv/(cp+0.J))
    #print(a)
    #print(b)

    e = np.array([a, b, cx, cy, theta])
    #if all(np.isreal(e[i]) is not True for i in e):
    #if np.isreal(e).all() is not True:
    if (np.all(np.isreal(e))):
        return e;    
    else:
        return np.array([0,0,0,0,0]);

def denormalize_ellipse_parameters(ne,Hn):
    e = np.empty_like(ne)
    #print(e)
    #print(ne)
    #print(Hn)
    e[0] = ne[0]/Hn[0,0]
    e[1] = ne[1]/Hn[1,1]
    e[2] = (ne[2] -Hn[0,2])/Hn[0,0]
    e[3] = (ne[3] -Hn[1,2])/Hn[2,2]
    e[4] = ne[4]
    
    return e 

def fit_ellipse_ransac (x, y, maximum_ransac_iterations, target_ellipse_radius, diviation):
    max_inliers = 0
    max_inlier_indices = []
    max_ellipse = []
    inliers_index = []
    ninliers = 0
    N = float('inf')
    ransac_iter = 0

#    if x.all() and y.all() is not True:
#         return

    ep_num = len(x)
    if ep_num < 5:
        print("Too few feature points")
        return
    #print(ep_num)
    #Normalize point coordinates
    cx = np.mean(x)
    cy = np.mean(y)
    mean_dist = np.mean(np.sqrt(np.square(x)+np.square(y)))
    dist_scale = np.sqrt(2)/mean_dist
    H = np.array([[dist_scale, 0, -dist_scale*cx],[0, dist_scale, -dist_scale*cy],[0,0,1]])
    nx = H[0,0]*x+H[0,2]
    ny = H[1,1]*y+H[1,2]
    #return nx, ny
    dist_thres = np.sqrt(3.84)*H[0,0]/4
    random_or_adaptive = 0

    #ep = np.transpose(np.array([[nx],[ny],[np.ones(len(nx))]]))
    ep = (np.array([nx, ny, np.ones(nx.size)])).T
    #print(ep)
    while (N > ransac_iter):
        if random_or_adaptive is 0:
            needed = 5.0
            #print(ep_num)
            available = ep_num
            #print(available)
            random_indices = np.zeros((needed, 1),dtype = np.int)

            while (needed > 0.0):
                if np.random.random(1) < np.float64(needed / available):
                    #print('iran')
                    random_indices[needed-1] = available-1
                    needed -= 1
                available -= 1
                #print(needed)
            nxi = nx[random_indices]
            nyi = ny[random_indices]
        else:
            nxi = nx[max_inlier_indices]
            nyi = ny[max_inlier_indices]
        #print(available)
        #print(nxi)
        #print(nyi)
        #A = np.transpose(np.array([[nxi*nxi],[nxi*nyi],[nyi*nyi],[nxi],[nyi],[np.ones(nxi.size)]], dtype = np.float32))
        A = np.transpose(np.array([nxi*nxi,nxi*nyi,nyi*nyi,nxi,nyi,np.ones(nxi.size)], dtype = np.float32))
        #print(A)

        #ua, sa, va = cv2.SVDecomp(A)
        ua, sa, va = np.linalg.svd(A)
        #vas = va.shape
        #vam = vas[0]
        #van = vas[1]
        vam, van = va.shape 
        nconic_par = va[:,-1]
        nconic_matrix = np.array([[nconic_par[0], nconic_par[1]/2, nconic_par[3]/2],
                                 [nconic_par[1]/2, nconic_par[2], nconic_par[4]/2],
                                 [nconic_par[3]/2, nconic_par[4]/2, nconic_par[5]]])
        #diserr = np.dot(ep,nconic_matrix)
        diserr = np.sum(np.multiply((np.dot(ep,nconic_matrix)),ep), axis=1)
        #print(diserr)
        #inliers_index = np.transpose(np.flatnonzero(np.abs(diserr) < dist_thres))
        inliers_index = np.nonzero(np.abs(diserr) < np.transpose(dist_thres))
        ninliers = len(inliers_index)
        #print(nconic_par)
        random_or_adaptive = 0
        if ninliers > max_inliers:
            nellipse_par = convert_conic_parameters_to_ellipse_parameters(nconic_par)
            #print(nellipse_par)
            if nellipse_par[0].all() > 0 and nellipse_par[1].all() > 0:
                ellipse_par = denormalize_ellipse_parameters(nellipse_par,H)
                er = np.divide(ellipse_par[0],ellipse_par[1])

                #er = ellipse_par[0] / ellipse_par[1]
                #print(ellipse_par[0])
                #print(ellipse_par[1])
              

                if target_ellipse_radius and diviation is not 0:
                    #print('imhere')
                    #print(er)
                    if (er > 0.75 and er < 1.34):
                        print('er')
                        print(er)
                    #if(np.divide(ellipse_par[0],target_ellipse_radius) < diviation):
                        #print('deviation')
                        #print(np.divide(ellipse_par[0],target_ellipse_radius))
                    #if(np.divide(ellipse_par[0],target_ellipse_radius) > 1/diviation):
                        #print('1/deviation')
                        #print(np.divide(ellipse_par[0],target_ellipse_radius))
                    #print(np.divide(ellipse_par[0], target_ellipse_radius))
                    
                    if (er > 0.75 and er < 1.34 and np.divide(ellipse_par[0],target_ellipse_radius) < diviation
                        and np.divide(ellipse_par[0],target_ellipse_radius) > 1/diviation 
                        and np.divide(ellipse_par[1],target_ellipse_radius) < diviation
                        and np.divide(ellipse_par[1],target_ellipse_radius) > 1/diviation):
                        max_inliers = ninliers
                        max_inlier_indices = inliers_index
                        max_ellipse = ellipse_par
                        N = np.log(1-0.99)/np.log(1-np.power((ninliers/ep_num),5)+np.spacing(1))
                        random_or_adaptive = 1
                    elif er > 0.75 and er < 1.34:
                        max_inliers = ninliers
                        max_inlier_indices = inliers_index
                        max_ellipse = ellipse_par
                        N = np.log(1-0.99)/np.log(1-np.power((ninliers/ep_num),5)+np.spacing(1))
                        random_or_adaptive = 1
        ransac_iter = ransac_iter + 1
        if ransac_iter > maximum_ransac_iterations:
            print("Maximum number of ransac iterations exeeded!")
            break

    return max_ellipse, max_inlier_indices, ransac_iter
        



# initial variables:

start_point = [-1, -1]
inliers_num = np.int16(0)
angle_step = np.int16(20)
pupil_edge_thresh = np.int16(6)
pupil_param = np.zeros(5)
edge_point = []
edge_intensity_diff = []
p = [0, 0]
edge = [0, 0]

def starburst_pupil_contour_detection (pupil_image, width, height, edge_thresh, N, minimum_candidate_features):

    global start_point, inliers_num, angle_step, pupil_edge_thresh, pupil_param, edge_point, edge_intensity_diff
    dis = np.int16(3)
    angle_spread = np.float64(180*3.1415926535897932384626433832795/180)
    loop_count = np.int16(0)
    angle_step = np.float64(2*3.1415926535897932384626433832795/N)
    new_angle_step = np.float64(0)
    angle_normal = np.float64(0)
    cx = np.float64(55)
    cy = np.float64(55)
    first_ep_num = np.int16(0)
    circleimage = cv2.imread('singletest.png',0)

    while (edge_thresh > 5 and loop_count <= 20):
        edge_intensity_diff = []
        edge_point = []
        #destroy_edge_point()
        while (len(edge_point) < minimum_candidate_features and edge_thresh > 5):

            edge_intensity_diff = []
            edge_point = []
            #print('doing edgy stuff')
            #print(edge_intensity_diff)
            #edge_intensity_diff = [(0)]
            #destroy_edge_point()
            locate_edge_points(pupil_image, width, height, 55, 55, 4, angle_step, 0, 2*3.1415926535897932384626433832795, edge_thresh)
            #print(edge_point)
            if (len(edge_point) < minimum_candidate_features):
                print('reduced threshold')
                edge_thresh -= 1
        if (edge_thresh <= 5):
            break;
        #print('test')
        #print(edge_intensity_diff)
        first_ep_num = len(edge_point)
        print(edge_point)
        for i in range(0, first_ep_num):
            edge = edge_point[i]
            #cv2.circle(circleimage, (edge[0], edge[1]), 2, (255,255,255), -1)
            angle_normal = np.arctan2(cy-edge[1], cx-edge[0])
            new_angle_step = angle_step*(edge_thresh*1.0/edge_intensity_diff[i])
            #print(angle_normal)
            #print(new_angle_step)
            locate_edge_points(pupil_image, width, height, edge[0], edge[1], 6, new_angle_step, angle_normal, angle_spread, edge_thresh)
        for i in range(0, len(edge_point)):
            edge = edge_point[i]
            cv2.circle(circleimage, (edge[0], edge[1]), 2, (255,255,255), -1)
            
        print(edge_point)

        loop_count += 1
        edge_mean = get_edge_mean()
        if(math.fabs(edge_mean[0]-cx) + math.fabs(edge_mean[1]-cy) < 10):
            break;
        cx = edge_mean[0]
        cy = edge_mean[1]

    if (loop_count > 10):
        destroy_edge_point()
        print('Error! Edge points did not converge')
        return;
    if (edge_thresh <= 5):
        destroy_edge_point()
        print('Error! Adaptive threshold too low')
        return;
    print(np.size(edge_point))
    ec = edge_point
    cv2.imshow('circleimage', circleimage)
    cv2.waitKey(0)
    return ec
    
    

def locate_edge_points(image, width, height, cx, cy, dis, angle_step, angle_normal, angle_spread, edge_thresh):

    global edge_point, edge_intensity_diff, p, edge
    p = [0, 0]
    edge = [0, 0]
    angle = np.float64(0)
    dis_cos = np.float64(0)
    dis_sin = np.float64(0)
    pixel_value1 = np.int16(0)
    pixel_value2 = np.int16(0)
    for angle in np.linspace(angle_normal-angle_spread/2+0.0001, angle_normal+angle_spread/2, ((angle_normal+angle_spread/2)-(angle_normal-angle_spread/2+0.0001))/angle_step):
        #print(angle)
        #print(a)
        #print(len(a))
        
        dis_cos = dis * np.cos(angle)
        dis_sin = dis * np.sin(angle)
        p[0] = math.floor(cx + dis_cos)
        p[1] = math.floor(cy + dis_sin)
        
        pixel_value1 = np.int16(image[p[1], p[0]])
        #print('wat')
        while (1):
            p[0] += dis_cos
            p[1] += dis_sin
            if (p[0] < 0 or p[0] >= width or p[1] < 0 or p[1] >= height):
                break;
            #print(p[0])
            #print(p[1])
            p[0] = math.floor(p[0])
            p[1] = math.floor(p[1])
            #print(p[0])
            #print(p[1])
            pixel_value2 = np.int16(image[p[1], p[0]])
            #print(pixel_value2)
            #print(pixel_value1)
            #print(pixel_value2 - pixel_value1)
            if (pixel_value2 - pixel_value1 > edge_thresh):
                edge = [0, 0]
                #print(p[0])
                #print(dis_cos/2)
                #raw_input('press enter to continue')
                edge[0] = np.int16(p[0] - dis_cos/2)
                edge[1] = np.int16(p[1] - dis_sin/2)
                edge_point.append(edge)
                #print(edge)
                edge_intensity_diff.append(pixel_value2 - pixel_value1)
                #print(pixel_value2)
                #print(pixel_value1)
                #print(pixel_value2 - pixel_value1)
                break;
            pixel_value1 = pixel_value2

def get_edge_mean():

    sumx = np.float64(0)
    sumy = np.float64(0)
    edge_mean = [0,0]

    for i in range(len(edge_point)):
        edge = edge_point[i]
        sumx += edge[0]
        sumy += edge[1]

    if (len(edge_point) != 0):
        edge_mean[0] = sumx / len(edge_point)
        edge_mean[1] = sumy / len(edge_point)
    else:
        edge_mean[0] = -1
        edge_mean[1] = -1
    return edge_mean;

def destroy_edge_point():

    #free stuff
    return;

def get_5_random_num (max_num, rand_num):

    rand_index = np.int16(0)
    r = np.int16(0)
    i = np.int16(0)
    is_new = 1

    if (max_num == 4):
        for i in range(5):
            rand_num[i] = i
        return;

    while (rand_index < 5):
        is_new = 1
        r = math.random(0, 5)
        for i in range(rand_index):
            if (r == rand_num[i]):
                is_new = 0
                break;
        if (is_new):
            rand_num[rand_index] = r
            rand_index += 1

def solve_ellipse (conic_param, ellipse_param):

    a = np.float64(conic_param[0])
    b = np.float64(conic_param[1])
    c = np.float64(conic_param[2])
    d = np.float64(conic_param[3])
    e = np.float64(conic_param[4])
    f = np.float64(conic_param[5])

    theta = np.float64(atan2(b, a-c)/2)

    ct = np.float64(cos(theta))
    st = np.float64(sin(theta))
    ap = a*ct*ct + b*ct*st + c*st*st
    cp = a*st*st - b*ct*st + c*ct*ct

    cx = (2*c*d - b*e) / (b*b - 4*a*c)
    cy = (2*a*e - b*d) / (b*b - 4*a*c)

    val = a*cx*cx + b*cx*cy + c*cy*cy
    scale_inv = val - f

    if (scale_inv/ap <= 0 or scale_inv/cp <= 0):
        print('le error, imaginary parameters')
        return 0;

    ellipse_param[0] = math.sqrt(scale_inv / ap)
    ellipse_param[1] = math.sqrt(scale_inv / cp)
    ellipse_param[2] = cx
    ellipse_param[3] = cy
    ellipse_param[4] = theta
    return 1;

def normalize_point_set(point_set, dis_scale, nor_center, num):

  sumx = np.float64(0)
  sumy = np.float64(0)
  sumdis = np.float64(0)
  edge = point_set
  i = np.int16(0)
  for i in range(num): 
    sumx += edge[0,0]
    sumy += edge[0,1]
    sumdis += math.sqrt(edge[0,0]*edge[0,0] + edge[0,1]*edge[0,1])
    edge += 1
  
  dis_scale = math.sqrt(2)*num/sumdis
  nor_center[0,0] = sumx*1.0/num
  nor_center[0,1] = sumy*1.0/num
  edge = point_set
  for i in range(num):
    edge_point_nor[i,0,0] = (edge[0,0] - nor_center[0,0])*dis_scale
    edge_point_nor[i,0,1] = (edge[0,1] - nor_center[0,1])*dis_scale
    edge += 1
  
  return edge_point_nor;


def normalize_edge_point(dis_scale, nor_center, ep_num):

  sumx = np.float64(0)
  sumy = np.float64(0)
  sumdis = np.float64(0)
  i = np.int16(0)
  for i in range(ep_num):
    edge = edge_point[i]
    sumx += edge[0, 0]
    sumy += edge[0, 1]
    sumdis += math.sqrt(edge[0, 0]*edge[0, 0] + edge[0, 1]*edge[0, 1])

  dis_scale = math.sqrt(2)*ep_num/sumdis
  nor_center[0,0] = sumx*1.0/ep_num
  nor_center[0,1] = sumy*1.0/ep_num
  edge_point_nor = np.float64(ep_num)
  for i in range(ep_num):
    edge = edge_point[i];
    edge_point_nor[i, 0, 0] = (edge[0,0] - nor_center[0,0])*dis_scale;
    edge_point_nor[i, 0, 1] = (edge[0,1] - nor_center[0,1])*dis_scale;
  
  return edge_point_nor;


def denormalize_ellipse_param(par, normailized_par, dis_scale, nor_center):

    par[0] = normailized_par[0] / dis_scale
    par[1] = normailized_par[1] / dis_scale
    par[2] = normailized_par[2] / dis_scale + nor_center[0,0]
    par[3] = normailized_par[3] / dis_scale + nor_center[0,1]


def pupil_fitting_inliers(pupil_image, width, height, return_max_inliers_num):

  i = np.int16(0)
  ep_num = np.int16(len(edge_point))
  dis_scale = np.float64(0)

  ellipse_point_num = np.int16(5)
  if (ep_num < ellipse_point_num):
    print('Error! 5 points are not enough to fit ellipse')
    return_max_inliers_num = 0
    return 0;

  edge_point_nor = normalize_edge_point(dis_scale, nor_center, ep_num)

  inliers_index = np.zeros(ep_num, dtype = int16)
  max_inliers_index = np.zeros(ep_num, dtype = int16)
  ninliers = np.int16(0)
  max_inliers = np.int16(0)
  sample_num = np.int16(1000)
  ransac_count = np.int16(0)
  dis_threshold = np.float64(math.sqrt(3.84)*dis_scale)
  dis_error = np.float64(0)
  
  rand_index = np.zeros(5, dtype = int16)
  A = np.zeros([6,6], dtype = double)
  M = np.int16(6)
  N = np.int16(6)
  for i in range(N):
    A[i,5] = 1
    A[5,i] = 0
  
  ppa = np.zeros(M)
  ppu = np.zeros(M)
  ppv = np.zeros(N)
  for i in range(M):
    ppa[i] = A[i]
    ppu[i] = np.zeros(N)
  
  for i in range(N):
    ppv[i] = np.zeros(N)
  
  pd = np.zeros(6) 
  min_d_index = np.int16(0)
  conic_par = np.zeros(6)
  ellipse_par = np.zeros(5)
  best_ellipse_par = np.zeros(5)
  ratio = np.float64(0)
  while (sample_num > ransac_count):
    get_5_random_num((ep_num-1), rand_index)
	
    for i in range(5):
      A[i,0] = edge_point_nor[rand_index[i],0,0] * edge_point_nor[rand_index[i],0,0]
      A[i,1] = edge_point_nor[rand_index[i],0,0] * edge_point_nor[rand_index[i],0,1]
      A[i,2] = edge_point_nor[rand_index[i],0,1] * edge_point_nor[rand_index[i],0,1]
      A[i,3] = edge_point_nor[rand_index[i],0,0]
      A[i,4] = edge_point_nor[rand_index[i],0,1]

    svd(M, N, ppa, ppu, pd, ppv)
    min_d_index = 0
    for i in range(1,N):
      if (pd[i] < pd[min_d_index]):
        min_d_index = i

    for i in range(N):
      conic_par[i] = ppv[i,min_d_index] 
                                       
    ninliers = 0
    inliers_index = np.zeros(ep_num, dtype = int16)
    for i in range(ep_num):
      dis_error = conic_par[0]*edge_point_nor[i,0,0]*edge_point_nor[i,0,0] + \
                  conic_par[1]*edge_point_nor[i,0,0]*edge_point_nor[i,0,1] + \
                  conic_par[2]*edge_point_nor[i,0,1]*edge_point_nor[i,0,1] + \
                  conic_par[3]*edge_point_nor[i,0,0] + conic_par[4]*edge_point_nor[i,0,1] + conic_par[5]
      if (math.abs(dis_error) < dis_threshold):
        inliers_index[ninliers] = i
        ninliers += 1
      
    if (ninliers > max_inliers):
      if (solve_ellipse(conic_par, ellipse_par)):
        denormalize_ellipse_param(ellipse_par, ellipse_par, dis_scale, nor_center)
        ratio = ellipse_par[0] / ellipse_par[1]
        if (ellipse_par[2] > 0 and ellipse_par[2] <= width-1 and ellipse_par[3] > 0 and ellipse_par[3] <= height-1 and
            ratio > 0.5 and ratio < 2):
          #memcpy(max_inliers_index, inliers_index, sizeof(int)*ep_num);
            for i in range(5):
                best_ellipse_par[i] = ellipse_par[i]
            max_inliers = ninliers
            sample_num = np.int16(math.log(np.float64(1-0.99))/math.log(1.0-math.pow(ninliers*1.0/ep_num, 5)))
        
      
    
    ransac_count += 1
    if (ransac_count > 1500):
      break;
    if (best_ellipse_par[0] > 0 and best_ellipse_par[1] > 0):
        for i in range(5):
            pupil_param[i] = best_ellipse_par[i]
    else:
        #memset(pupil_param, 0, sizeof(pupil_param));
        max_inliers = 0
        #free(max_inliers_index);
        max_inliers_index = NULL

    #for i in range(M):
        #free(ppu[i])
        #free(ppv[i])
    #free(ppu);
    #free(ppv);
    #free(ppa);

    #free(edge_point_nor);
    #free(inliers_index);
    return_max_inliers_num = max_inliers
    return max_inliers_index;

def ellipse_direct_fit(xy):
    #centroid = np.mean(xy)
    

    #centroid = np.array([np.mean(xy[0,:]),np.mean(xy[1,:])])
    print ("------- \n")
    print (xy)

    centroid = np.mean(xy, axis = 0)
    print (centroid)


    print(xy[:,0])
    print(xy[:,1])

    D1 = np.array([np.square(xy[:,0]-centroid[0]), (xy[:,0]-centroid[0])*(xy[:,1]-centroid[1]),
                   np.square(xy[:,1]-centroid[1])]).T
    D2 = np.array([xy[:,0]-centroid[0], xy[:,1]-centroid[1], np.ones((np.size(xy, axis = 0),1))]).T
    #S1 = np.dot(np.transpose(D1),D1)
    #S2 = np.dot(np.transpose(D1),D2)
    #S3 = np.dot(np.transpose(D2),D2)
    #S1 = np.transpose(D1)*D1
    #S2 = np.transpose(D1)*D2
    #S3 = np.transpose(D2)*D2
    
    S1 = np.dot(D1.T,D1)
    #print(S1)
    S2 = np.dot(D1.T,D2)
    #print(S2)
    S3 = np.dot(D2.T,D2)
    #print(S3)

    #T = np.dot(-np.linalg.inv(S3),np.transpose(S2))
    T = -np.dot(np.linalg.inv(S3),S2.T)
    M = S1 + np.dot(S2,T)
    Mm = np.array([M[2,:]/2,-M[1,:],M[0,:]/2])

    eval, evec = np.linalg.eig(Mm)
    cond = 4*evec[0,:]*evec[2,:]-np.square(evec[1,:])
    #A1 = evec[:,np.nonzero(cond>0)]
    A1 = evec[:,cond.T>0]
    #A = np.array([[A1],[np.dot(T,A1)]])
    A = np.array([A1, np.dot(T,A1)])
    A4 = A[3]-2*A[0]*centroid[0]-A[1]*centroid[1]
    A5 = A[4]-2*A[2]*centroid[1]-A[1]*centroid[0]
    A6 = (A[5]+A[0]*np.square(centroid[0])+A[2]*np.square(centroid[1])
          +A[1]*centroid[0]*centroid[1]-A[3]*centroid[0]-A[4]*centroid[1])
    A[3] = A4
    A[4] = A5
    A[5] = A6
    A = A/np.linalg.norm(A)

    return A


image = cv2.imread('singletest.png', 0)
imH = np.size(image, 0)
print(imH)
imW = np.size(image, 1)
print(imW)
#print(image[66,116])
ec = starburst_pupil_contour_detection (image, imW, imH, 7, 10, 8)
print(len(ec))
ecx = np.array(np.empty(len(ec)))
ecy = np.array(np.empty_like(ecx))
for x in range(0, len(ec)):
    ecx[x] = ec[x][0]
    ecy[x] = ec[x][1]
ellipse, inliers, ransac_iter = fit_ellipse_ransac(ecx, ecy, 1000, 10, 1.5)
if len(ellipse) is 0 or ransac_iter >= 10000:
    print("No ellipse found")
else:
    c = ellipse_direct_fit(np.array([np.transpose(ecx[inliers]),np.transpose(ecy[inliers])]))

    ellipse = converst_conic_parameters_to_ellipse_parameters(c)

#e_shortAxis = ellipse[0]
#e_longAxis = ellipse[1]
#e_centerx = ellipse[2]
#e_centery = ellipse[3]
    e_angle = ellipse[4]
    e_center = np.array([ellipse[2],ellipse[3]])
    e_axes = np.array([ellipse[0],ellipse[1]])

    image_ellipse = cv2.ellipse(image, e_center, e_axes, e_angel, 0, 360, (0,0,0), 1)
    cv2.imshow('circleimage', image_ellipse)
    cv2.waitKey(0)
                      



















    
    
