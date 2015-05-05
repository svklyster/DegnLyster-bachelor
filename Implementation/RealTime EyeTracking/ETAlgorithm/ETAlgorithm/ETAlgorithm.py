import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import sys


# ransac funtion, go!

def convert_conic_parameters_to_ellipse_parameters(c):
    
    c.shape = (6,)
   
    theta = np.arctan2(c[1], c[0]-c[2])/2
    ct = np.cos(theta)
    st = np.sin(theta)
    ap = c[0]*ct*ct+c[1]*ct*st+c[2]*st*st
    cp = c[0]*st*st-c[1]*ct*st+c[2]*ct*ct

    #T = np.array([np.transpose([c[0], c[1]/2]),np.transpose([c[1]/2, c[2]])])
    T = np.array((np.transpose([c[0],c[1]/2]), np.transpose([c[1]/2,c[2]])))
    #T.shape = (2,2)
    try:
        t = np.linalg.solve(-(2*T),np.transpose(np.array([c[3],c[4]])))
    except np.linalg.linalg.LinAlgError as err:
        pass
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
                er = np.divide(ellipse_par[0],ellipse_par[1]).real
                print(er)
                #er = ellipse_par[0] / ellipse_par[1]
                #print(ellipse_par[0])
                #print(ellipse_par[1])
              

                if target_ellipse_radius and diviation is not 0:
                    #print('imhere')
                    #print(er)
                    #if (er > 0.75 and er < 3.34):
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
inliers_num = np.int16(0)
angle_step = np.int16(20)
pupil_edge_thresh = np.int16(6)
pupil_param = np.zeros(5)
edge_point = []
edge_intensity_diff = []
p = [0, 0]
edge = [0, 0]


def remove_corneal_reflection (imagePntr, threshPntr, sx, sy, windowSize, biggest_crr, crx, cry, crr, imgW, imgH):
    crar = np.int16(-1)
    #crx = cry = crar = np.int16(-1)

    angle_delta = np.float32(0.01745329251994329576923690768489)

    angle_num = np.int16(2*3.1415926535897932384626433832795/angle_delta)
    angle_array = np.zeros(angle_num)
    sin_array = np.zeros(angle_num)
    cos_array = np.zeros(angle_num)

    for x in range(angle_num):
        angle_array[x] = x*angle_delta
        sin_array[x] = math.sin(angle_array[x])
        cos_array[x] = math.cos(angle_array[x])

    [crx, cry, crar] = locate_corneal_reflection(imagePntr, threshPntr, sx, sy, windowSize, np.int16(biggest_crr/2.5), crx, cry, crar, imgW, imgH)
    crr = fit_circle_radius_to_corneal_reflection(imagePntr, crx, cry, crar, np.int16(biggest_crr/2.5), sin_array, cos_array, angle_num, imgW, imgH)
    #crr = np.int16(2.5*crr)
    imagePntr = interpolate_corneal_reflection(imagePntr, crx, cry, crr, sin_array, cos_array, angle_num, imgW, imgH)
    #cv2.imshow('after', imagePntr)
    #cv2.waitKey(0)
    return crx, cry, crar
    # free stuff

def locate_corneal_reflection (imagePntr, threshPntr, sx, sy, windowSize, biggest_crar, crx, cry, crr, imW, imH):

    r = np.int16((windowSize-1)/2)
    startx = np.int16(max(sx-r, 0))
    endx = np.int16(min(sx+r, imW-1))
    starty = np.int16(max(sy-r, 0))
    endy = np.int16(min(sy+r, imH-1))

    imageROI = imagePntr
    threshROI = threshPntr
    #imageROI= imagePntr[130:230, 400:1000]
    #imageROI = imagePntr[starty:endy-starty+1, startx:endx-startx+1]
    #threshROI = threshPntr[130:230, 400:1000]
    #threshROI = threshPntr[starty:endy-starty+1, startx:endx-startx+1]

    min_value = np.float64(0)
    max_value = np.float64(0)
    min_loc = ()
    max_loc = ()
    [min_value, max_value, min_loc, max_loc] = cv2.minMaxLoc(imageROI)

    threshold = np.int16(0)
    i = np.int16(0)
    #CvSeq contour=null
    #CvMemStorage storage = cvCreateMemStorage(0)
    # scores = malloc(sizeof(double)*((int)max_value+1))
    #memset(scores, 0, sizeof(double)*((int)max_value+1))
    scores = np.zeros(np.int16(max_value+1))
    area = np.int16(0)
    max_area = np.int16(0)
    sum_area = np.int16(0)
    for threshold in range(np.int16(max_value), 0, -1):
        ret, threshROI = cv2.threshold(imageROI, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('current thresh', threshROI)
        #cv2.waitKey(0)
        contour, hierarchy = cv2.findContours(threshROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #cnt = contour[0]
        max_area = 0
        sum_area = 0
        max_contour = contour
        for contourCount in range(len(contour)):
            if (contour != 0):
                area = len(contour) + np.int16(cv2.contourArea(contour[contourCount]))
                #print(np.int16(cv2.contourArea(contour[contourCount])))
                sum_area += area
                if(area > max_area):
                    max_area = area
                    #print(max_area)
                    max_contour = contour
        if (sum_area-max_area > 0):
            scores[threshold-1] = max_area / (sum_area-max_area)
        else:
            continue
        
        if (scores[threshold-1] - scores[threshold] < 0):
            #print('this never happens!')
            #print(max_area)
            crar = np.int16(math.sqrt(max_area / 3.1415926535897932384626433832795))
            #print(crar)
            sum_x = np.int16(0)
            sum_y = np.int16(0)
            for i in range(len(max_contour)):
                point = max_contour[i]
                #print(point)

                sum_x += point[0,:,0]
                sum_y += point[0,:,1]

            #print(sum_x)
            #print(sum_y)
            #print(len(max_contour))
            crx = sum_x/len(max_contour)
            cry = sum_y/len(max_contour)
            break

    # free stuff
    #cv2.drawContours(imagePntr, contour, -1, (0,255,0),5)
    
    if (crar > biggest_crar):
        cry = crx = -1
        crar = -1

    return crx, cry, crar;

def fit_circle_radius_to_corneal_reflection (imagePntr, crx, cry, crar, biggest_crar, sin_array, cos_array, array_len, imW, imH):

    #print(crx)
    #print(cry)
    #print(crar)
    crar = 8
    if (crx == -1 or cry == -1 or crar == -1):
        return -1;

    ratio = np.zeros(biggest_crar-crar+1)
    #print(ratio)
    i = np.int16(0)
    r = np.int16(0)
    r_delta = np.int16(1)
    x = np.int16(0)
    y = np.int16(0)
    x2 = np.int16(0)
    y2 = np.int16(0)
    sum1 = np.float64(0)
    sum2 = np.float64(0)

    for r in range(crar, biggest_crar):
        sum1 = 0
        sum2 = 0
        count = 0
        for i in range(array_len):
            x = (int)(crx + (r+r_delta)*cos_array[i])
            y = (int)(cry + (r+r_delta)*sin_array[i])
            x2 = (int)(crx + (r-r_delta)*cos_array[i])
            y2 = (int)(cry + (r+r_delta)*sin_array[i])
            count += 1
            if ((x >= 0 and y >= 0 and x < imW and y < imH) and
                (x2 >= 0 and y2 >= 0 and x2 < imW and y2 < imH)):
                sum1 += imagePntr[y, x]
                sum2 += imagePntr[y2, x2]
        ratio[r-crar] = sum1/sum2

        if (r - crar >= 2):
            if (ratio[r-crar-2] < ratio[r-crar-1] and ratio[r-crar] < ratio[r-crar-1]):
                #free(ratio)
                return r-1;

    #free(ratio)
    #print stuff
    return crar;

def interpolate_corneal_reflection (imagePntr, crx, cry, crr, sin_array, cos_array, array_len, imW, imH):

    #print(crx)
    #print(cry)
    #print(crr)
    if (crx == -1 or cry == -1 or crr == -1):
        return;

    if (crx-crr < 0 or crx + crr >= imW or cry-crr < 0 or cry+crr >= imH):
        return;

    i = np.int16(0)
    r  = np.int16(0)
    r2 = np.int16(0)
    x  = np.int16(0)
    y  = np.int16(0)
    perimeter_pixel = np.zeros(array_len, dtype=np.uint8)
    sum1 = np.int32(0)
    pixel_value = np.int16(0)
    avg = np.float64(0)
    count = 0
    
    for i in range(array_len):
        x = np.int16(crx + crr * cos_array[i])
        #print(x)
        y = np.int16(cry + crr * sin_array[i])
        #print(y)
        perimeter_pixel[i] = imagePntr[y,x]
        sum1 += perimeter_pixel[i]

    avg = sum1/array_len
    #print(avg)
    for r in range(crr):
        r2 = crr-r
        for i in range(array_len):
            x = (int)(crx + r*cos_array[i])
            #print(x)
            y = (int)(cry + r*sin_array[i])
            #print(y)
            #print((r2/crr)*avg + (r/crr)*perimeter_pixel[i])
            imagePntr[y,x] = np.uint8((r2/crr)*avg + (r/crr)*perimeter_pixel[i])
            
            count = count + 1;
  
    return imagePntr;



def starburst_pupil_contour_detection (pupil_image, width, height, cx, cy, edge_thresh, N, minimum_candidate_features):

    global inliers_num, angle_step, pupil_edge_thresh, pupil_param, edge_point, edge_intensity_diff
    
    dis = np.int16(3)
    angle_spread = np.float64(180*3.1415926535897932384626433832795/180)
    loop_count = np.int16(0)
    angle_step = np.float64(2*3.1415926535897932384626433832795/N)
    new_angle_step = np.float64(0)
    angle_normal = np.float64(0)
    cx = np.float64(cx)
    cy = np.float64(cy)
    first_ep_num = np.int16(0)
    #circleimage = pupil_image

    while (edge_thresh > 5 and loop_count <= 20):
        edge_intensity_diff = []
        edge_point = []
        #destroy_edge_point()
        while (len(edge_point) < minimum_candidate_features and edge_thresh > 5):

            edge_intensity_diff = []
            edge_point = []
            locate_edge_points(pupil_image, width, height, cx, cy, 4, angle_step, 0, 2*3.1415926535897932384626433832795, edge_thresh)
            #print(edge_point)
            if (len(edge_point) < minimum_candidate_features):
                print('reduced threshold')
                edge_thresh -= 1
        if (edge_thresh <= 5):
            break;
        #print('test')
        #print(edge_intensity_diff)
        first_ep_num = len(edge_point)
        #print(edge_point)
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
            #cv2.circle(circleimage, (edge[0], edge[1]), 2, (255,255,255), -1)
            
        #print(edge_point)

        loop_count += 1
        edge_mean = get_edge_mean()
        if(math.fabs(edge_mean[0]-cx) + math.fabs(edge_mean[1]-cy) < 10):
            break;
        cx = edge_mean[0]
        cy = edge_mean[1]

    if (loop_count > 10):
        destroy_edge_point()
        sys.exit('Error! Edge points did not converge')
        return;
    if (edge_thresh <= 5):
        destroy_edge_point()
        sys.exit('Error! Adaptive threshold too low')
        return;
    print(np.size(edge_point))
    ec = edge_point
    #cv2.imshow('circleimage', circleimage)
    #cv2.waitKey(0)
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
    
    centroid = np.mean(xy, axis = 0)

    D1 = np.array([np.square(xy[:,0]-centroid[0]), (xy[:,0]-centroid[0])*(xy[:,1]-centroid[1]),
                   np.square(xy[:,1]-centroid[1])]).T
    D2 = np.array([xy[:,0]-centroid[0], xy[:,1]-centroid[1], np.ones((np.size(xy, axis = 0),1))]).T
    S1 = np.dot(D1.T,D1)
    #print(S1)
    S2 = np.dot(D1.T,D2)
    #print(S2)
    S3 = np.dot(D2.T,D2)
    #print(S3)

    #T = np.dot(-np.linalg.inv(S3),np.transpose(S2))
    try:
        T = -np.dot(np.linalg.inv(S3),S2.T)
    except: 
        sys.exit("Singular value matrix")
    M = S1 + np.dot(S2,T)
    Mm = np.array([M[2,:]/2,-M[1,:],M[0,:]/2])

    eval, evec = np.linalg.eig(Mm)
    cond = np.multiply(4*evec[0,:],evec[2,:])-np.square(evec[1,:])
    #A1 = evec[:,np.nonzero(cond>0)]
    A1 = evec[:,cond.T>0]
    #A = np.array([[A1],[np.dot(T,A1)]])
    #A = np.array([[A1],  [np.dot(T,A1)]])
    #A = np.concatenate(A1, np.dot(T,A1))
    A = np.vstack((A1, np.dot(T,A1)))
    A4 = A[3]-2*A[0]*centroid[0]-A[1]*centroid[1]
    A5 = A[4]-2*A[2]*centroid[1]-A[1]*centroid[0]
    A6 = (A[5]+A[0]*np.square(centroid[0])+A[2]*np.square(centroid[1])
          +A[1]*centroid[0]*centroid[1]-A[3]*centroid[0]-A[4]*centroid[1])
    A[3] = A4
    A[4] = A5
    A[5] = A6
    A = A/np.linalg.norm(A)

    return A


image = cv2.imread('test.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in eyes:
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray_thresh = gray[y:y+h, x:x+w]

    #image = cv2.GaussianBlur(image, (5,5), 0)

    #start_point = [-1, -1]
    inliers_num = np.int16(0)
    angle_step = np.int16(20)
    pupil_edge_thresh = np.int16(6)
    pupil_param = np.zeros(5)
    edge_point = []
    edge_intensity_diff = []
    p = [0, 0]
    edge = [0, 0]

    #forventet midterpunkt
    sy = 100
    sx = 100

    windowSize = 199

    imH = np.size(roi_gray, 0)

    imW = np.size(roi_gray, 1)

    
    crx, cry, crr = remove_corneal_reflection(roi_gray, roi_gray_thresh, sy, sx, windowSize, 100, 2, 2, -2, imW, imH)

    #cv2.circle(roi_gray, (crx, cry), crr, (255,255,255), 1)
    #cv2.line(roi_gray, (crx-5, cry), (crx+5, cry), (255,255,255), 1)
    #cv2.line(roi_gray, (crx, cry-5), (crx, cry+5), (255,255,255), 1)
    cv2.imshow('image', roi_gray)
    cv2.waitKey(0)


    ec = starburst_pupil_contour_detection (roi_gray, imW, imH, crx, cry, 7, 10, 4)

    ecx = np.array(np.empty(len(ec)))
    ecy = np.array(np.empty_like(ecx))
    for x in range(0, len(ec)):
        ecx[x] = ec[x][0]
        ecy[x] = ec[x][1]
    ellipse, inliers, ransac_iter = fit_ellipse_ransac(ecx, ecy, 1000, 10, 1.5)
    if len(ellipse) is 0 or ransac_iter >= 10000:
        sys.exit("No ellipse found")
    else:
        c = ellipse_direct_fit(np.array([ecx[inliers], ecy[inliers]]).T)
        ellipse = convert_conic_parameters_to_ellipse_parameters(c)
        e_angle = int(ellipse[4]).real*57.2957795 
        e_center = (ellipse[2],ellipse[3])
        e_axes = (ellipse[0],ellipse[1])
        cv2.ellipse(roi_gray, e_center, e_axes, e_angle, 0, 360, (255,255,255), 1)
        cv2.imshow('Ellipse', roi_gray)
        cv2.waitKey(0)
#        cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#cv2.imshow('image', image)
#cv2.waitKey(0)        


