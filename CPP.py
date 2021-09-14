import math
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from numpy.lib.function_base import append
#import cv2 as cv


def calculate_polygons(startx, starty, endx, endy, radius):
    
    # https://www.techforwildlife.com/blog/2019/1/29/calculating-a-drone-cameras-image-footprint
    h = math.sqrt(2)
    fov = math.radians(90)
    r = 1
    
    A = (2*h*math.tan(fov/2))/math.sqrt(1+r**2)
    B = (2*r*h*math.tan(fov/2))/math.sqrt(1+r**2)
    
    if r>1:
        radius = A/2
    else:
        radius = B/2
    
    sl = radius
    
    # calculate coordinates of the hexagon points
    p = sl * 0.5
    b = sl * math.cos(math.radians(30))
    w = b * 2
    h = 2 * sl
    
    # offset start and end coordinates by hex widths and heights to guarantee coverage     
    startx = startx - w/2
    starty = starty - h/2
    endx = endx + w/2
    endy = endy + h/2

    origx = startx
    origy = starty
    
    # offsets for moving along and up rows
    xoffset = b
    yoffset = 3 * p

    polygons = []
    row = 1
    counter = 0

    while starty < endy:
        if row % 2 == 0:
            startx = origx + xoffset
        else:
            startx = origx
        while startx < endx:
            p1x = startx
            p1y = starty + p
            p2x = startx
            p2y = starty + (3 * p)
            p3x = startx + b
            p3y = starty + h
            p4x = startx + w
            p4y = starty + (3 * p)
            p5x = startx + w
            p5y = starty + p
            p6x = startx + b
            p6y = starty
            poly = [(p1x, p1y),(p2x, p2y),(p3x, p3y),(p4x, p4y),(p5x, p5y),(p6x, p6y),(p1x, p1y)]
            polygons.append(poly)
            counter += 1
            startx += w
        starty += yoffset
        row += 1
    return polygons


def get_centroids(grid):

    cent = []
    
    for i in range(len(grid)):
        tempx = []
        tempy = []
        
        for j in range(6):
            tempx.append(grid[i][j][0])
            tempy.append(grid[i][j][1])
        sx = sum(tempx)
        sy = sum(tempy)
        sx = sx/6
        sy = sy/6
        cent.append([sx,sy])
    return cent


def get_image():
    
    #file1 = open("File1.txt", "w")
    #file2 = open("File2.txt", "w")
    
    image2 = img.imread('image.png')
    #pix = [[0]*len(image[0])]*len(image[1])
    #for i in range(len(image[0])):
        #file1.write(str(i)+':')
        #for j in range(len(image[1])):
            #pix[i][j] = sum(image[i][j])
            #out_arr = np.array_str(image[i][j])
            #file1.write(str(j)+':'+out_arr+'\t')
        #file1.write(out_arr+'\n')
            
    image = image2.copy()

    for i in range(len(image[0])):
        #file2.write(str(i)+':')
        for j in range(len(image[1])):
            image[j][i] = image2[63-i][j]
            #out_arr = np.array_str(image2[i][j])
            #file2.write(str(j)+':'+out_arr+'\t')
        #file2.write(out_arr+'\n')
        
    #plt.imshow(image)
    #plt.show()
    
    return image
    
    #file1.close()
    #file2.close()
    

def get_waypoints(centroid,image):

    gx,gy,gc = [],[],[]
    wx,wy = [],[]
    
    for i in range(len(centroid)):
        px = int(round(centroid[i][0]))
        py = int(round(centroid[i][1]))
        if px>63 or py>63:
            continue
        if image[px][py][0]+image[px][py][1]+image[px][py][2] < 1:
            #print(px)
            #print(py)
            for j in range(7):
                gx.append(grid[i][j][0])
                gy.append(grid[i][j][1])
            #gc.append(centroid[i])
            wx.append(centroid[i][0])
            wy.append(centroid[i][1])
            
    
    #figure, axis = plt.subplots(2, 1)
    #axis[0].plot(gx, gy)
    plt.plot(gx, gy)
    plt.show() 
    
    #axis[1].imshow(image)
    #plt.imshow(image)
    #cv.waitKey(0)
    #plt.show()
            
    plt.scatter(wx, wy, marker= ".")
    plt.show() 



grid = calculate_polygons(0,0,64,64,1)
#print(grid[0])

x,y = [],[]
for i in range(len(grid)):
    for j in range(7):
        x.append(grid[i][j][0])
        y.append(grid[i][j][1])
        
#plt.plot(x,y)
#plt.show()

centroid = get_centroids(grid)
#print(centroid)

image = get_image()

#print(image[0][0][0].astype(int)<<16)
#print(image[0][0][1].astype(int)<<8)
#print(image[0][0][2])

#RGBint = (image[30][20][0].astype(int)<<16) + (image[30][20][1].astype(int)<<8) + image[30][20][2].astype(int)
#print(RGBint)

get_waypoints(centroid,image)

