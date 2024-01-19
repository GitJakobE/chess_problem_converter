import math
import cv
import numpy as np

templates = ['wpawn.png', 'whorse.png', 'bpawn.png', 'bhorse.png']


image = cv.imread("chessboard.png")
imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

for template_name in templates:
    template = cv.imread(template_name)
    templateGray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    w, h = templateGray.shape[::-1]
    res = cv.matchTemplate(imageGray, templateGray, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        print("Found at ", pt);
        cv.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    
    
edges = cv.Canny(imageGray,50,150,apertureSize = 3)

lines = cv.HoughLines(edges,1,np.pi/180,200)
singleDegInRadians = math.pi/180.0;
for i in range(len(lines)):
    for rho,theta in lines[i]:
        if abs(theta)>5.0*singleDegInRadians and abs(theta - math.pi/2.0)>5.0*singleDegInRadians and abs(theta-math.pi)>5.0*singleDegInRadians and abs(theta - 3.0*math.pi/2.0)>5.0*singleDegInRadians:
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)

cv.imwrite('output.png', image)

