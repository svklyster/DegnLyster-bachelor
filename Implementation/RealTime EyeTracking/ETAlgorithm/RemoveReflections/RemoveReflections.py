import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def remove_corneal_reflection (imagePntr, threshPntr, sx, sy, windowSize, biggest_crr, crx, cry, crr):
    crar = np.int16(-1)
    crx = cry = crar = np.int16(-1)

    angle_delta = np.float32(0.01745329251994329576923690768489)

    angle_num = np.int16(2*3.1415926535897932384626433832795/angle_delta)
    angle_array = np.zeros(angle_num)
    sin_array = np.zeros(angle_num)
    cos_array = np.zeros(angle_num)

    for x in range(angle_num):
        angle_array[x] = x*angle_delta
        sin_array[x] = math.sin(angle_array[x])
        cos_array[x] = math.cos(angle_array[x])

    [crx, cry, crar] = locate_corneal_reflection(imagePntr, threshPntr, sx, sy, windowSize, np.int16(biggest_crr/2.5), crx, cry, crar, 239, 136)
    crr = fit_circle_radius_to_corneal_reflection(imagePntr, crx, cry, crar, np.int16(biggest_crr/2.5), sin_array, cos_array, angle_num, 239, 136)
    crr = np.int16(2.5*crr)
    imagePntr = interpolate_corneal_reflection(imagePntr, crx, cry, crr, sin_array, cos_array, angle_num, 239, 136)
    cv2.imshow('after', imagePntr)
    cv2.waitKey(0)
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
                print(np.int16(cv2.contourArea(contour[contourCount])))
                sum_area += area
                if(area > max_area):
                    max_area = area
                    print(max_area)
                    max_contour = contour
        if (sum_area-max_area > 0):
            scores[threshold-1] = max_area / (sum_area-max_area)
        else:
            continue
        
        if (scores[threshold-1] - scores[threshold] < 0):
            print('this never happens!')
            print(max_area)
            crar = np.int16(math.sqrt(max_area / 3.1415926535897932384626433832795))
            print(crar)
            sum_x = np.int16(0)
            sum_y = np.int16(0)
            for i in range(len(max_contour)):
                point = max_contour[i]
                print(point)

                sum_x += point[0,:,0]
                sum_y += point[0,:,1]

            print(sum_x)
            print(sum_y)
            print(len(max_contour))
            crx = sum_x/len(max_contour)
            cry = sum_y/len(max_contour)
            break

    # free stuff
    #cv2.drawContours(imagePntr, contour, -1, (0,255,0),5)
    plt.imshow(threshROI, 'gray')
    plt.show()

    if (crar > biggest_crar):
        cry = crx = -1
        crar = -1

    if (crx != -1 and cry != -1):

        #crx += startx
        #cry += starty
        print(crx)
        print(cry)

    return crx, cry, crar;

def fit_circle_radius_to_corneal_reflection (imagePntr, crx, cry, crar, biggest_crar, sin_array, cos_array, array_len, imW, imH):

    #print(crx)
    #print(cry)
    #print(crar)
    crar = 4
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
            #print(crx)
            #print(r)
            #print(r_delta)
            #print(cos_array[i])
            cv2.waitKey(0)
            y = (int)(cry + (r+r_delta)*sin_array[i])
            x2 = (int)(crx + (r-r_delta)*cos_array[i])
            y2 = (int)(cry + (r+r_delta)*sin_array[i])
            count += 1
            if ((x >= 0 and y >= 0 and x < imW and y < imH) and
                (x2 >= 0 and y2 >= 0 and x2 < imW and y2 < imH)):
                #print("nope")
                sum1 += imagePntr[y, x]
                sum2 += imagePntr[y2, x2]
        #print(count)
        #print(sum1)
        #print(sum2)
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
    sum1 = np.int16(0)
    pixel_value = np.int16(0)
    avg = np.float64(0)
    count = 0
    print('so far so good')
    
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
    print(imagePntr)
    return imagePntr;
    #free(perimeter_pixel)

image = cv2.imread('singletest.png',0)  
thresh_image = cv2.imread('singletest.png',0)
image = cv2.GaussianBlur(image, (5,5), 0)

remove_corneal_reflection(image, thresh_image, 200, 100, 199, 100, 2, 2, -2)

