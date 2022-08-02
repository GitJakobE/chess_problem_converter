import math
import cv2
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pdf2image import convert_from_path


templates = ['wpawn.png', 'whorse.png', 'bpawn.png', 'bhorse.png']
source_image = "data/Toft-00-0252.png" if len(sys.argv)<2 else sys.argv[1];
debug =True
# Main function 
def main():
    #import an pdf and convert its content to pngs
    convert_pdf_to_pngs("data/Toft-00/Toft-00-0000.pdf")

    # import, convert to grayscale and binarize
    image = cv2.imread(source_image)
    blue_channel = image[:,:,2]
    red_channel = image[:,:,0]
    green_channel = image[:,:,1]
    if(debug==True):
        cv2.imwrite('out/blue_image.png', blue_channel)
        cv2.imwrite('out/red_image.png', red_channel)
        cv2.imwrite('out/green_image.png', green_channel)

    blue_img = np.zeros(image.shape)
    blue_img[:,:,0] = blue_channel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    retval, image_gray = cv2.threshold(blue_channel, 128, 256, cv2.THRESH_BINARY)
    if(debug==True):
        cv2.imwrite('out/grayimage.png', image_gray)
    # fig, ax = plt.subplots()
    # im = ax.imshow(image)
    # # plt.show()
    cv2.imshow('image', blue_img)
    # Determine lines using hough transform
    horizontalLines, verticalLines = findHorizontalVerticalLines(image_gray);
    # debug: draw_lines(image_gray, horizontalLines);        
    
    # Determine angle of lines and use to derotate image
    angle = findAverageRotation(horizontalLines, verticalLines);
    angle_deg = 180.0*angle/math.pi;
    derotated_image_gray = rotate_image(image_gray, angle_deg)

    # Locate the positions of the 'templates'
    template_locations = locate_templates(derotated_image_gray);

    # Locate lines again on derotated image to find bounding box
    horizontalLines, verticalLines = findHorizontalVerticalLines(derotated_image_gray);
    bounding_box = find_bounding_box(horizontalLines, verticalLines);

    bb_image_gray = image
    cv2.rectangle(bb_image_gray, (bounding_box[0],bounding_box[1]), (bounding_box[2],bounding_box[3]), (0,0,255), 2)
    # cv2.imshow('image', bb_image_gray)
    cv2.imwrite('output.png', derotated_image_gray)
    fig, ax = plt.subplots()
    im = ax.imshow(bb_image_gray)
    plt.show()
    # Setup the 8x8 grid and find out which cells are occupied by which templates
    grid=setup_grid();
    grid = fill_templates_in_grid(grid, bounding_box, template_locations)
    # Output grid as JSON
    print(json.dumps(grid));
    
    # Draw output
    draw_locations(derotated_image_gray, template_locations);
    cv2.imwrite('output.png', derotated_image_gray)

# Locate points where the templates appear in image_gray
def locate_templates(image_gray):
    locations = [];
    for template_name in templates:
        template = cv2.imread(template_name)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]
        res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        locations.append({'name':template_name, 'w':w, 'h':h, 'positions':np.where( res >= threshold) }); 
        
    return locations;    
        

def convert_pdf_to_pngs(filename: str)->None:
    pages = convert_from_path(filename, 500)
    n = 0
    filename = str.replace(filename, ".pdf", "")
    for page in pages:
        page.save(f"{filename}--{n}.png")
        n +=1

# Draw template locations
def draw_locations(target_image, locations):
    for template in locations:
        for pt in zip(*template['positions'][::-1]):
            cv2.rectangle(target_image, pt, (pt[0] + template['w'], pt[1] + template['h']), (0,255,255), 2)

    
# Locate all lines < 15 degrees rotated
def findHorizontalVerticalLines(imageGray):
    edges = cv2.Canny(imageGray,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    singleDegInRadians = math.pi/180.0;
    maxRotationDegrees = 15.0;
    maxRotationRadians =  maxRotationDegrees/180.0 * math.pi;
    horizontalLines = [];
    verticalLines = [];

    for i in range(len(lines)):
        for rho,theta in lines[i]:
            if abs(theta)<maxRotationRadians or abs(theta - math.pi)<maxRotationRadians:
                verticalLines.append(lines[i]);
            if abs(theta - math.pi/2.0)<maxRotationRadians or abs(theta - 3.0*math.pi/2.0)<maxRotationRadians:
                horizontalLines.append(lines[i]);
    return horizontalLines, verticalLines;        


# Rotate an image by angle degrees
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


# Returns points for hough line defined by rho and theta
def points_for_line(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    return x0,y0,x1,y1,x2,y2;
    

# Draw hough lines on targetImage            
def draw_lines(targetImage, lines):
    for i in range(len(lines)):
        for rho,theta in lines[i]:
            x0,y0,x1,y1,x2,y2 = points_for_line(rho,theta);

            cv2.line(targetImage,(x1,y1),(x2,y2),(0,0,255),2)
 
# Move angle into [-PI,PI] 
def unifyAngle(a):
    if a > math.pi/2.0:
        return a - math.pi; 
    return a;    
 
 
# Determines average rotation from a set of horizontal and vertical lines
def findAverageRotation(horizontalLines, verticalLines):
    total_lines = len(horizontalLines) + len(verticalLines);
    if total_lines<1:
        return 0;
    avg_theta = 0.0;    
    for l in horizontalLines:
        avg_theta += unifyAngle(l[0][1] - math.pi/2.0);
    for l in verticalLines:
        avg_theta += unifyAngle(l[0][1]);
    
    return avg_theta / total_lines;
    

# return BB of lines (min_x, min_y, max_x, max_y)    
def find_bounding_box(horizontalLines, verticalLines):
    min_x = 1024;
    max_x = 0;    
    min_y = 1024;
    max_y = 0;    
    for l in horizontalLines:
        _,_,_,y,_,_ = points_for_line(l[0][0],l[0][1]);
        min_y = min(min_y, y);
        max_y = max(max_y, y);
    for l in verticalLines:
        _,_,x,_,_,_ = points_for_line(l[0][0],l[0][1]);
        min_x = min(min_x, x);
        max_x = max(max_x, x);
    return min_x, min_y, max_x, max_y;
    
    
# Return unfilled grid [8][8] of empty string
def setup_grid():
    grid=[];
    for y in range(8):
        grid_row=[];
        for x in range(8):
            grid_row.append("");
        grid.append(grid_row);
    return grid;            


# Go through each cell and set the content of the cell to the (last) template the was found in the cell
def fill_templates_in_grid(grid, bounding_box, template_locations):
    x1=bounding_box[0];
    w=bounding_box[2]-bounding_box[0];
    y1=bounding_box[1];
    h=bounding_box[3] - bounding_box[1];
    for y in range(8):
        for x in range(8):
            gx1=x1+w/8.0*x;
            gy1=y1+h/8.0*y;
            gx2=gx1+w/8.0;
            gy2=gy1+h/8.0;
            for template in template_locations:
                for pt in zip(*template['positions'][::-1]):
                    if pt[0]>gx1 and pt[0]<gx2 and pt[1]>gy1 and pt[1]<gy2:
                        grid[y][x]=template['name'];
    return grid;

# Run the program          
if __name__ == "__main__":
    main()