
import numpy as np
import cv2
import math
import sys
import EyeTracking as et
import cv2.cv as cv

def Track(frame, e_center, last_eyes, calData, runVJ):

    def convert_conic_parameters_to_ellipse_parameters(c):
        #c = c.real
        if c.size is not 6:
            return np.array([0,0,0,0,0])

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
            return np.array([0,0,0,0,0])
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
            return np.array([0,0,0,0,0])

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
        max_inliers = np.float64(0)
        max_inlier_indices = []
        max_ellipse = []
        inliers_index = []
        ninliers = np.float64(0)
        N = float('inf')
        ransac_iter = 0

    #    if x.all() and y.all() is not True:
    #         return
        #x = np.array([5,4,3,4,5,4,3,4,5])
        #y = np.array([4,5,4,3,4,3,4,5,4])
    
        ep_num = np.float64(len(x))
        #print "ep_num:"
        #print ep_num
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
            nconic_par = va[-1,:]
            nconic_matrix = np.array([[nconic_par[0], nconic_par[1]/2, nconic_par[3]/2],
                                     [nconic_par[1]/2, nconic_par[2], nconic_par[4]/2],
                                     [nconic_par[3]/2, nconic_par[4]/2, nconic_par[5]]])
            #diserr = np.dot(ep,nconic_matrix)
            diserr = np.sum(np.multiply((np.dot(ep,nconic_matrix)),ep), axis=1)
            #print(diserr)
            #inliers_index = np.transpose(np.flatnonzero(np.abs(diserr) < dist_thres))
            inliers_index = np.nonzero(np.abs(diserr) < np.transpose(dist_thres))
            ninliers = len(inliers_index[0])
            #print(nconic_par)
            random_or_adaptive = 0
            if ninliers > max_inliers:
                nellipse_par = convert_conic_parameters_to_ellipse_parameters(nconic_par)
                #print(nellipse_par)
                if nellipse_par[0].all() > 0 and nellipse_par[1].all() > 0:
                    ellipse_par = denormalize_ellipse_parameters(nellipse_par,H)
                    er = np.divide(ellipse_par[0],ellipse_par[1]).real
                    #print(er)
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
                    
                        if (er > 0.85 and er < 1.15 and np.divide(ellipse_par[0],target_ellipse_radius) < diviation
                            and np.divide(ellipse_par[0],target_ellipse_radius) > 1/diviation 
                            and np.divide(ellipse_par[1],target_ellipse_radius) < diviation
                            and np.divide(ellipse_par[1],target_ellipse_radius) > 1/diviation):
                            max_inliers = ninliers
                            max_inlier_indices = inliers_index
                            max_ellipse = ellipse_par
                            N = np.log(1.0-0.99)/np.log(1.0-np.power((ninliers/ep_num),5)+np.spacing(1))
                            random_or_adaptive = 1
                        #elif er > 0.85 and er < 1.15:
                        #    max_inliers = ninliers
                        #    max_inlier_indices = inliers_index
                        #    max_ellipse = ellipse_par
                        #    N = np.log(1-0.99)/np.log(1-np.power((ninliers/ep_num),5)+np.spacing(1))
                        #    random_or_adaptive = 1
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

    def remove_corneal_reflection (imagePntr, threshPntr, sx, sy, windowSize, biggest_crr, crx, cry, crr, imgW, imgH, e_center):


        [crx, cry, contour] = locate_corneal_reflection(imagePntr, threshPntr, sx, sy, windowSize, np.int16(biggest_crr/2.5), crx, cry, imgW, imgH, e_center)
        
        contourCenter, contourRadius = cv2.minEnclosingCircle(np.concatenate((contour[0], contour[1]), axis=0))
        #crr = fit_circle_radius_to_corneal_reflection(imagePntr, crx, cry, crar, np.int16(biggest_crr/2.5), imW, imH) 
        #crr = int(2.5*crr)
        cv2.equalizeHist(imagePntr, imagePntr)
        maxIntensity = 255.0
        x = np.arange(maxIntensity)
        phi = 1
        theta = 1
        roi_gray = (maxIntensity/phi)*(imagePntr/(maxIntensity/theta))**0.5
        roi_gray = np.array(imagePntr, dtype=np.uint8)
        
        imagePntr = interpolate_corneal_reflection(imagePntr, int(contourCenter[0]), int(contourCenter[1]), contourRadius*3, contour, imW, imH, e_center)

        
        #contourRect = cv2.boundingRect(np.concatenate((contour[0], contour[1]), axis=0))

        if imagePntr is None:
            return None, None

        #cv2.imshow('image', imagePntr)
        #cv2.waitKey(0)

        return crx, cry

    def locate_corneal_reflection (imagePntr, threshPntr, sx, sy, windowSize, biggest_crar, crx, cry, imW, imH, e_center):

        #r = np.int16((windowSize-1)/2)
        #startx = np.int16(max(sx-r, 0))
        #endx = np.int16(min(sx+r, imW-1))
        #starty = np.int16(max(sy-r, 0))
        #endy = np.int16(min(sy+r, imH-1))

        imageROI = imagePntr
        threshROI = threshPntr

        min_value = np.float64(0)
        max_value = np.float64(0)
        min_loc = ()
        max_loc = ()
        [min_value, max_value, min_loc, max_loc] = cv2.minMaxLoc(imageROI)

        threshold = np.int16(0)
        i = np.int16(0)
        scores = np.zeros(np.int16(max_value+1))
        area = np.int16(0)
        max_area = np.int16(0)
        sum_area = np.int16(0)
        for threshold in range(np.int16(max_value), 0, -1):
            ret, threshROI = cv2.threshold(imageROI, threshold, 255, cv2.THRESH_BINARY)

            contour, hierarchy = cv2.findContours(threshROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            max_area = 0
            sum_area = 0
            indx = 0
            
      

            max_contour = contour

            for contourCount in range(len(contour)):
                #if (contour[contourCount] != 0):
                area = len(contour[contourCount]) + np.int16(cv2.contourArea(contour[contourCount]))
                sum_area += area
                if(area >= max_area):
                    max_area = area
                    max_contour[indx] = contour[contourCount]
                    indx += 1
            if (sum_area-max_area > 0):
                scores[threshold-1] = max_area / (sum_area-max_area)
            else:
                continue
        
            if (scores[threshold-1] - scores[threshold] < 0):
                #crar = int(np.sqrt(max_area / np.pi))

                sum_x = np.int16(0)
                sum_y = np.int16(0)

                if len(max_contour) > 2 and e_center is not None:
                    dist = []
                    for i in range(len(max_contour)):
                        con_mean = np.mean(contour[i], axis=0)
                 
                        dist.append(np.sqrt(np.square(e_center[0] - con_mean[0][0]) + np.square(e_center[1] - con_mean[0][1])))
               
                    nContour = []
                    indx = np.nanargmin(dist)
                    nContour.append(max_contour[indx])
                    dist[indx] = np.nan
                    indx = np.nanargmin(dist)
                    nContour.append(max_contour[indx])
        
                elif len(max_contour) is 2:
                    nContour = max_contour  
        
                else:
                    return None

                for i in range(len(nContour)):
                    point = nContour[i]

                    sum_x += point[0,:,0]
                    sum_y += point[0,:,1]
                
                
                crx = sum_x/len(nContour)
                cry = sum_y/len(nContour)
                break
        #cv2.imshow("Woldsoa", threshROI)
        return crx, cry, nContour;

    def fit_circle_radius_to_corneal_reflection (imagePntr, crx, cry, crar, biggest_crar, imW, imH):

        #print(crx)
        #print(cry)
        #print(crar)
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

        angles = int(2*np.pi/(np.pi/180));

        for r in range(crar, biggest_crar):
            sum1 = 0
            sum2 = 0
            count = 0
            for i in range(angles):
                x = int(crx + (r+r_delta)*np.cos(i))
                y = int(cry + (r+r_delta)*np.sin(i))
                x2 = int(crx + (r-r_delta)*np.cos(i))
                y2 = int(cry + (r+r_delta)*np.sin(i))
                count += 1
                if ((x >= 0 and y >= 0 and x < imW and y < imH) and
                    (x2 >= 0 and y2 >= 0 and x2 < imW and y2 < imH)):
                    sum1 += imagePntr[y, x]
                    sum2 += imagePntr[y2, x2]
            ratio[r-crar] = sum1/sum2

            if (r - crar >= 2):
                if (ratio[r-crar-2] < ratio[r-crar-1] and ratio[r-crar] < ratio[r-crar-1]):
                    return r-1;

        return crar;


    def interpolate_corneal_reflection (imagePntr, crx, cry, crr, nContour, imW, imH, e_center):

        if crx-crr < 0 or crx+crr >= imW or cry-crr < 0 or cry+crr >= imH:
            return None

        perimeter_pixel = []
        angles = int(2*np.pi/(np.pi/180));
        sum = 0

        cv2.imshow('circleimage', imagePntr)
        cv2.waitKey(0)

        for i in range(angles):
            x = int(crx+crr*np.cos(i))
            y = int(cry+crr*np.sin(i))
            perimeter_pixel.append(imagePntr[x,y])
            sum += perimeter_pixel[i]

        havg = sum/angles/10

     

        cv2.circle(imagePntr, (crx,cry), int(crr), int(havg), -1)

        #for r in range(crr):
        #    r2 = crr-r+1
        #    for i in range(angles):
        #        x = int(crx + r*np.cos(i))
        #        y = int(crx + r*np.sin(i))
        #        imagePntr[x,y] = np.int((np.float(r2)/np.float(crr))*avg + (np.float(r)/np.float(crr))*perimeter_pixel[i])
        ##for i in range(len(nContour)):
        #    rect[i] = cv2.boundingRect(nContour[i]) #returnerer x,y(top left corner), widht, height
        #    #fittedRect[i] = [rect[i][0]-2*rect[i][2],rect[i][1]-2*rect[i][3], rect[i][2]*4, rect[i][3]*4]
        #    c_center = (int(rect[i][0]+np.ceil(rect[i][2]/2)), int(rect[i][1]+np.ceil(rect[i][3]/2)))
        #    c_radius = np.max([rect[i][2]*3,rect[i][3]*3])
        #    #k = 5
        #    #try:
        #    #    color = np.array([imagePntr[c_center[0]-c_radius-k, c_center[1]],imagePntr[c_center[0], c_center[1]-c_radius-k],imagePntr[c_center[0]+c_radius+k, c_center[1]], imagePntr[c_center[0], c_center[i]+c_radius+k]],dtype = np.float)

        #    #    #color[i] = np.array([imagePntr[fittedRect[i][1]-k,fittedRect[i][0]-k],imagePntr[fittedRect[i][1]-k,fittedRect[i][0]+fittedRect[i][2]*2+k],imagePntr[fittedRect[i][1]+fittedRect[i][3]*2+k, fittedRect[i][0]-k], imagePntr[fittedRect[i][1]+fittedRect[i][3]*2+k, fittedRect[i][0]+fittedRect[i][2]*2+k]])
        #    #except:
        #    #    color = (255,255,255)
            
        #    #min_indx = int(np.argmin(color))
        #    #min_color = color[min_indx]
        #    #min_color = (0,0,0)
        #    min_color = int(np.min(imagePntr))
        #    #color = (min_color,min_color,min_color)
        #    cv2.circle(imagePntr, c_center, c_radius, min_color, -1)
        cv2.imshow('circleimage', imagePntr)
        cv2.waitKey(0)
        return imagePntr;



    def starburst_pupil_contour_detection (pupil_image, width, height, cx, cy, pupil_edge_thresh, N, minimum_candidate_features):

        global edge_point, edge_intensity_diff
    
        dis = np.int16(3)
        angle_spread = np.float64(180*3.1415926535897932384626433832795/180)
        loop_count = np.int16(0)
        angle_step = np.float64(2*3.1415926535897932384626433832795/N)
        new_angle_step = np.float64(0)
        angle_normal = np.float64(0)
        cx = np.float64(cx)
        cy = np.float64(cy)
        first_ep_num = np.int16(0)
        circleimage = pupil_image
        edge_thresh = pupil_edge_thresh

        while (edge_thresh > 5 and loop_count <= 20):
            epx = []
            epy = []
            while (len(epx) < minimum_candidate_features and edge_thresh > 5):
                epx, epy, epd = locate_edge_points(pupil_image, width, height, cx, cy, dis, angle_step, 0, 2*3.1415926535897932384626433832795, edge_thresh)
                if (len(epx) < minimum_candidate_features):
                    #print('reduced threshold')
                    edge_thresh -= 1
            
            #if (edge_thresh <= 5):
            #    edge_thresh = pupil_edge_thresh
            #    while (len(epx) < minimum_candidate_features and edge_thresh > 5):
            #        epx, epy, epd = locate_edge_points(pupil_image, width, height, cx, cy, dis, angle_step, 0, 2*3.1415926535897932384626433832795, edge_thresh, contour)
            #        #tepx, tepy, tepd = locate_edge_points(pupil_image, width, height, contour[0]+contour[2]/2, contour[1]+contour[3]/2, dis, angle_step, 0, 2*np.pi, edge_thresh, contour)
            #        #epx = np.hstack((epx, tepx))
            #        #epy = np.hstack((epy, tepy))
            #        #epd = np.hstack((epd, tepd))
            #        if len(epx) < minimum_candidate_features:
            #            edge_thresh -= 1
            first_ep_num = len(epx)
            #print(edge_point)
            for i in range(0, first_ep_num):
                edge = [int(epx[i]),int(epy[i])]
                cv2.circle(circleimage, (edge[0], edge[1]), 2, (255,255,255), -1)
                angle_normal = np.arctan2(cy-epy[i], cx-epx[i])
                new_angle_step = angle_step*(edge_thresh*1.0/epd[i])
                tepx, tepy, tepd = locate_edge_points(pupil_image, width, height, cx, cy, dis, new_angle_step, angle_normal, angle_spread, edge_thresh)
                epx = np.hstack([epx, tepx])
                epy = np.hstack([epy, tepy])
            cv2.imshow('firstround',circleimage)
            cv2.waitKey(0)
            for i in range(0, len(epx)):
                edge = [int(epx[i]),int(epy[i])]
                cv2.circle(circleimage, (edge[0], edge[1]), 1, (255,255,255), -1)
            cv2.imshow('secondround',circleimage)
            cv2.waitKey(0)

            loop_count += 1
            tcx = np.mean(epx)
            tcy = np.mean(epy)

            #edge_mean = np.mean(edge_point,0)
            #edge_mean = get_edge_mean()
            if(np.abs(tcx-cx) + np.abs(tcy-cy) < 10):
                break;
            cx = tcx
            cy = tcy
        #cv2.imshow('circleimage', circleimage)
        #cv2.waitKey(0)
        if (loop_count > 10):
            #destroy_edge_point()
            print('Error! Edge points did not converge')
            return None, None
        if (edge_thresh <= 5):
            #destroy_edge_point()
            print('Error! Adaptive threshold too low')
            return None, None
        #ec = [epx, epy]
        
        return epx, epy
    
    

    def locate_edge_points(image, width, height, cx, cy, dis, angle_step, angle_normal, angle_spread, edge_thresh):

        epx = []
        epy = []
        dir = []
        ep_num = 0
        #p = np.zeros([4,2])
        p = [0, 0]
        #edge = [0, 0]

        #reflectionKeepoutX = 1.5*contour[2]
        #reflectionKeepoutY = 1.5*contour[3]
        #reflectionX = contour[0]+contour[2]/2
        #reflectionY = contour[1]+contour[3]/2

        #reflectionXmax = reflectionX + reflectionKeepoutX
        #reflectionXmin = reflectionX - reflectionKeepoutX
        #reflectionYmax = reflectionY + reflectionKeepoutY
        #reflectionYmin = reflectionY - reflectionKeepoutY

        angle = np.float64(0)
        angle = np.float64(0)
        dis_cos = np.float64(0)
        dis_sin = np.float64(0)
        pixel_value1 = np.int16(0)
        pixel_value2 = np.int16(0)

        alpha = 0.01

        for angle in np.arange(angle_normal-angle_spread/2+0.0001, angle_normal+angle_spread/2, angle_step):
            
            dis_cos = dis * np.cos(angle)
            dis_sin = dis * np.sin(angle)
            p[0] = math.floor(cx + dis_cos)
            p[1] = math.floor(cy + dis_sin)
            
            pixel_ema = 0.0
            pixel_sum = 0
            count = 1
            pixel_value1 = np.int16(image[p[1], p[0]])
            
            while (1):
                p[0] += dis_cos
                p[1] += dis_sin
                if (p[0] < 0 or p[0] >= width or p[1] < 0 or p[1] >= height):
                    break

                #if (p[0] <= reflectionXmax and p[0] >= reflectionXmin and p[1] <= reflectionYmax
                #    and p[1] >= reflectionYmin):
                #    break

                #else:

                p[0] = math.floor(p[0])
                p[1] = math.floor(p[1])

                pixel_value2 = np.int16(image[p[1], p[0]])
                #pixel_sum += pixel_value2
                #pixel_ema = pixel_sum / count

                pixel_ema = alpha*pixel_value2 + (1-alpha)*pixel_ema

                if (pixel_value2 - pixel_value1 > edge_thresh or pixel_value2 - pixel_ema > edge_thresh):
                #if (pixel_value2 - pixel_ema > edge_thresh): 
                    epx.append(np.int16(p[0] - dis_cos/2))
                    epy.append(np.int16(p[1] - dis_sin/2))
                    #print(p[0])
                    #print(p[1])
                    dir.append(pixel_value2 - pixel_value1)

                    break;
                pixel_value1 = pixel_value2
                count += 1.3
        
        return epx, epy, dir

            #step = 2
            #p[1,:] = [np.round(cx+dis*np.cos(angle)),np.round(cy+dis*np.sin(angle))]
            #if p[1,1] >= height or p[1,1] < 0 or p[1,0] >= width or p[1,0] < 0:
            #    continue
            #while 1:
            #    p[0] = [np.round(cx+step*dis*np.cos(angle)), np.round(cy+step*dis*np.sin(angle))]
            #    if p[0,1] >= height or p[0,1] < 0 or p[0,0] >= width or p[0,0] < 0:
            #        break
            #    d = (image[p[0,1],p[0,0]]-image[p[1,1],p[1,0]])*2*np.square((255-image[p[1,1], p[1,0]])/255)
            #    if d >= edge_thresh:
            #        ep_num += 1
            #        epx.append((p[0,0]+p[1,0])/2)
            #        epy.append((p[0,1]+p[1,1])/2)
            #        dir.append(d)
            #        break
            #    if p[2,1] > 0:
            #        d2 = (image[p[0,1],p[0,0]]-image[p[2,1],p[2,0]])*2*np.square((255-image[p[2,1], p[2,0]])/255)
            #        if d2 >= edge_thresh:
            #            ep_num += 1
            #            epx.append((p[0,0]+p[2,0])/2)
            #            epy.append((p[0,1]+p[2,1])/2)
            #            dir.append(d2)
            #            break
            #    else:
            #        d2 = 0

            #    if p[3,1] > 0:
            #        d3 = (image[p[0,1],p[0,0]]-image[p[3,1],p[3,0]])*2*np.square((255-image[p[3,1], p[3,0]])/255)
            #        if d3 >= edge_thresh:
            #            ep_num += 1
            #            epx.append((p[0,0]+p[3,0])/2)
            #            epy.append((p[0,1]+p[3,1])/2)
            #            dir.append(d3)
            #            break
            #    else:
            #        d3 = 0
                
            #    p[3,:] = p[2,:]
            #    p[2,:] = p[1,:]
            #    p[1,:] = p[0,:]
            #    step += 1


        #global edge_point, edge_intensity_diff, p, edge
        #p = [0, 0]
        #edge = [0, 0]
        #angle = np.float64(0)
        #dis_cos = np.float64(0)
        #dis_sin = np.float64(0)
        #pixel_value1 = np.int16(0)
        #pixel_value2 = np.int16(0)
        #for angle in np.linspace(angle_normal-angle_spread/2+0.0001, angle_normal+angle_spread/2, ((angle_normal+angle_spread/2)-(angle_normal-angle_spread/2+0.0001))/angle_step):
        #    #print(angle)
        #    #print(a)
        #    #print(len(a))
        
        #    dis_cos = dis * np.cos(angle)
        #    dis_sin = dis * np.sin(angle)
        #    p[0] = math.floor(cx + dis_cos)
        #    p[1] = math.floor(cy + dis_sin)
        
        #    pixel_value1 = np.int16(image[p[1]-1, p[0]-1])
        #    #print('wat')
        #    while (1):


                #p[0] += dis_cos
                #p[1] += dis_sin
                #if (p[0] < 0 or p[0] >= width or p[1] < 0 or p[1] >= height):
                #    break;
                ##print(p[0])
                ##print(p[1])
                #p[0] = math.floor(p[0])
                #p[1] = math.floor(p[1])
                ##print(p[0])
                ##print(p[1])
                #pixel_value2 = np.int16(image[p[1], p[0]])
                ##print(pixel_value2)
                ##print(pixel_value1)
                ##print(pixel_value2 - pixel_value1)
                #if (pixel_value2 - pixel_value1 > edge_thresh):
                #    edge = [0, 0]
                #    #print(p[0])
                #    #print(dis_cos/2)
                #    #raw_input('press enter to continue')
                #    edge[0] = np.int16(p[0] - dis_cos/2)
                #    edge[1] = np.int16(p[1] - dis_sin/2)
                #    edge_point.append(edge)
                #    #print(edge)
                #    edge_intensity_diff.append(pixel_value2 - pixel_value1)
                #    #print(pixel_value2)
                #    #print(pixel_value1)
                #    #print(pixel_value2 - pixel_value1)
                #    break;
                #pixel_value1 = pixel_value2


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
          et.ReturnError("Singular value matrix")
          return 0
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


    #image = cv2.imread('test.png')
    image = frame
    #print("Test1")
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    #cv2.imshow('image', hist)
    #cv2.waitKey(0)
    if runVJ is True:
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 10, minSize = (100,100) )
   
    else:
        ###TEST###
        #lastW = 200
        #lastH = 200
        ##########
        eyes = last_eyes
        #eyes = np.array([int((e_center[0]-lastW/2).real), int((e_center[1]-lastH/2).real), int(lastW), int(lastH)])
        #eyes = np.array([(e_center[0]-lastW/2).real, (e_center[1]-lastH/2).real, lastW, lastH], dtype = np.int)  
        ## Er oejet hoejre eller venstre (meget grov kode)
        if eyes is None:
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 10, minSize = (100,100) )
    e_found = [0,0]
    new_e_center = [0,0]*2

    for (x,y,w,h) in eyes:
        if x > np.size(image, 1)/2+20:
            #antager at oejet er venstre hvis det ligger i venstre del af billedet
            e_orientation = "Right"
            e_found[0] = 1
            try:
                if e_center is not None and e_center[0][0] > 0:
                    eye_center = e_center[0]
                else:
                    eye_center = [100,100]
            except:
                eye_center = [100,100]
        else:
            e_orientation = "Left"
            e_found[1] = 1
            try:
                if e_center is not None and e_center[1][0] > 0:
                    eye_center = e_center[1]
                else:
                    eye_center = [100,100]
            except:
                eye_center = [100,100]
        #print(e_orientation)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_thresh = gray[y:y+h, x:x+w]

        #cv2.imshow('image', roi_gray)
        #cv2.waitKey(0)

        #roi_gray = cv2.GaussianBlur(roi_gray, (5,5), 0)
        #roi_gray_thresh = cv2.GaussianBlur(roi_gray_thresh, (5,5), 0)
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
        if eye_center is not None and eye_center[0] > 0:
            sx = eye_center[0]
            sy = eye_center[1]
        else:
            sy = 100
            sx = 100

        windowSize = 199

        imH = np.size(roi_gray, 0)

        imW = np.size(roi_gray, 1)

        
    
        crx, cry, = remove_corneal_reflection(roi_gray, roi_gray_thresh, sy, sx, windowSize, 20, 2, 2, -2, imW, imH, eye_center)

        if crx is None or cry is None:
            et.ReturnError("Error with corneal reflections")
            return

       

        #cv2.imshow('newImage0',roi_gray)
        #cv2.waitKey(0)

        #y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

        # Decrease intensity such that
        # dark pixels become much darker, 
        # bright pixels become slightly dark 
        #newImage1 = (maxIntensity/phi)*(roi_gray/(maxIntensity/theta))**2
        #newImage1 = np.array(newImage1,dtype=np.uint8)

        #cv2.imshow('newImage1',newImage1)
        #cv2.waitKey(0)

        #cv2.circle(roi_gray, (crx, cry), crr, (255,255,255), 1)
        #cv2.line(roi_gray, (crx-5, cry), (crx+5, cry), (255,255,255), 1)
        #cv2.line(roi_gray, (crx, cry-5), (crx, cry+5), (255,255,255), 1)
        #cv2.imshow('image', roi_gray)
        #cv2.waitKey(0)
    

        epx, epy = starburst_pupil_contour_detection (roi_gray, imW, imH, crx, cry, 40, 6, 6)
        if epx is None:
            et.ReturnError("Starburst threshold too low")
            return

        #ecx = np.array(np.empty(len(ec)))
        #ecy = np.array(np.empty_like(ecx))
        #for x in range(0, len(ec)):
        #    ecx[x] = ec[x][0]
        #    ecy[x] = ec[x][1]
        ellipse, inliers, ransac_iter = fit_ellipse_ransac(epx, epy, 1000, 10, 1.5)
        if len(ellipse) is 0 or ransac_iter >= 10000:
            et.ReturnError("Maximum ransac iterations exceeded")
            return
           
        
        #print "ransac_iter:"
        #print ransac_iter
        
        else:
            if (np.float64(inliers[0].size)/np.float64(epx.size)) < 0.3 or inliers[0].size < 20:
                et.ReturnError("Asuming false positive")
                return
            c = ellipse_direct_fit(np.array([epx[inliers], epy[inliers]]).T)
            ellipse = convert_conic_parameters_to_ellipse_parameters(c)
            e_angle = int(ellipse[4]).real*57.2957795 
            e_axes = (ellipse[0],ellipse[1])
            if e_orientation is "Right":
                new_e_center[0] = (ellipse[2],ellipse[3])
                cv2.ellipse(roi_gray, new_e_center[0], e_axes, e_angle, 0, 360, (255,255,255), 1)
                cv2.imshow('Ellipse', roi_gray)
                cv2.waitKey(0)  
            else:
                new_e_center[1] = (ellipse[2],ellipse[3])
                cv2.ellipse(roi_gray, new_e_center[1], e_axes, e_angle, 0, 360, (255,255,255), 1)
                cv2.imshow('Ellipse', roi_gray)
                cv2.waitKey(0)  
           
            #return (crx,cry), e_center, ("NoTrigger")
            
            #gaze_vector = ((e_center[0]-crx).real, (e_center[1]-cry).real)
            gaze_vector = (1,1)
            #####CALIBRATION#####
            if calData is not None:

                coef_vector = (np.square(gaze_vector[1]), np.square(gaze_vector[0]))
                screen_pos = (coef_vector*calData.x, coef_vector*calData.y)+calData.offset
            
            
            
            #et.LastRunInfo(e_center, eyes)
            et.EyesFound(e_found)
            et.PackWithTimestamp(e_center, gaze_vector, ("NoTrigger"))
            
        et.LastRunInfo(new_e_center, eyes)
        return




    
