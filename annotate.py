# importing the module 
import cv2
import numpy as np
import copy
import os

global p1, p2, z
global coordinate_p1
global midline_eq, edge_1_eq, edge_2_eq # Tuples of gradient, y-intercept

f=open('text_speed.txt','w')

buckets = []
dots_midline = []
flow = None
result = None
normalize_meters = 0.5 # This is the 10 meters
temp_approx_meters_p1 = 0.5 # Basically, allow the user to put any scale rather than forcing them to put something of 10m. Hardcoded to 2 rn.
temp_approx_meters_p2 = 0.5 # Same as above but for the other end

f.write(str(normalize_meters) + "\n")
list1 = []
img = cv2.imread('image.jpg', 1)
cached = copy.deepcopy(img)
test = 1 # First is to take the first 10m. Second is to take the last 10m. Third is to take the whole road and draw a line midpoint. Then it is reset.

def click_event(event, x, y, flags, params):
    global list1
    global img
    global cached
    global p1, p2, z
    global test
    global result
    global midline_eq, edge_1_eq, edge_2_eq
    global dots_midline, dots_midline, buckets, f

    # To get starting point 10m
    if len(list1) == 2 and test == 1:
        test +=1

        cv2.line(img, tuple(list1[0]), tuple(list1[1]), (0,0,0), 3)
        cv2.imshow('image', img)

        p1 = pytha(list1[0], list1[1]) * (normalize_meters / temp_approx_meters_p2)

        # Empty list if we want another square next time
        list1 = []

    # To get end point 10m
    elif len(list1) == 2 and test == 2:
        test += 1

        cv2.line(img, tuple(list1[0]), tuple(list1[1]), (0,0,0), 3)
        cv2.imshow('image', img)
        cv2.imwrite('Lined.jpg', img)

        p2 = pytha(list1[0], list1[1]) * (normalize_meters / temp_approx_meters_p2)

        list1 = []
    
    # To extract the whole road
    elif len(list1) == 4 and test == 3:
        test += 1

        # Makes polygon
        pts = np.array(list1)

        temp_img = cached
        mask = np.zeros(temp_img.shape).astype(img.dtype)
        cv2.fillPoly(mask, [pts], (255,255,255))
        # We can use this mask for any image with same camera view.
        # result=cv2.bitwise_and(temp_img, mask)
        cv2.imwrite("mask.jpg", mask)


        cv2.polylines(img, [pts], True, (0,255,255))
        cv2.imshow('image', img)

        # Draw Midpoint line
        point1 = list1[0]
        point2 = list1[1]
        point3 = list1[2]
        point4 = list1[3]

        midline_p1 = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
        midline_p2 = ((point3[0] + point4[0]) // 2, (point3[1] + point4[1]) // 2)

        cv2.line(img, midline_p1, midline_p2, (0,255,255), 1)
        cv2.imshow('image', img)

        # Extract its gradient and y intercept
        gradient_midline = gradient(midline_p1, midline_p2)
        y_intercept_midline = y_intercept(midline_p1, gradient_midline)
        midline_eq = (gradient_midline, y_intercept_midline) # Tuple with (gradient, y-intercept)

        # Extract length edges gradient and y intercept
        gradient_edge_1 = gradient(point1, point4)
        y_intercept_edge_1 = y_intercept(point1, gradient_edge_1)
        edge_1_eq = (gradient_edge_1, y_intercept_edge_1)

        gradient_edge_2 = gradient(point2, point3)
        y_intercept_edge_2 = y_intercept(point2, gradient_edge_2)
        edge_2_eq = (gradient_edge_2, y_intercept_edge_2)

        # Test equations
        print("\nMY LINES")
        print("Midline:", midline_eq, "Edge 1:", edge_1_eq, "Edge 2:", edge_2_eq)
        print("")

        # Using midpoint lines, we get the z value which is total pixels in the line.
        z = pytha(midline_p1, midline_p2)

        create_dots(midline_p1, midline_p2, result) # Have to draw the 10m marks now.
        # make_buckets(img)
        # find_speed(img)

        list1 = []
        dots_midline = []

    # Left button click for coordinates
    if event == cv2.EVENT_LBUTTONDOWN: 
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y)

        list1.append([x,y])
  
        # displaying the coordinates 
        # on the image window 
        # font = cv2.FONT_HERSHEY_SIMPLEX 
        # cv2.putText(img, str(x) + ',' +
        #             str(y), (x,y), font, 
        #             1, (255, 0, 0), 2) 
        cv2.imshow('image', img)

    # # Right button click to clear 
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     f.close()
    #     f=open('cords.txt','w')
    #     list1 = []
    #     buckets = []
    #     dots_midline = []
    #     test = 1
    #     img = copy.deepcopy(cached)
    #     cv2.imshow("image", img)

# Gotta do the 10m marks. Saves the dots in a list of tuple with coordinates and pixels. [((x1,y1), pixels), ...]
def create_dots(midline_p1, midline_p2, result):
    global p1, p2, z, img, coordinate_p1, dots_midline, f, midline_eq

    line_distance = z

    # Get the dimensions of midline triangle
    # print("Midlines:", midline_p1, midline_p2)
    horizontal_diff = abs(midline_p1[0] - midline_p2[0])
    vertical_diff = abs(midline_p1[1] - midline_p2[1])
    # print("Horizontal Difference:", horizontal_diff, vertical_diff)

    # From the starting point, I want 10m
    # Get ratio
    ratio_p1 = p1 / line_distance
    # Get pixels moved
    horizontal_moved_p1 = ratio_p1 * horizontal_diff
    vertical_moved_p1 = ratio_p1 * vertical_diff
    # print(horizontal_moved_p1)
    # print(vertical_moved_p1)
    # Get new coordinates
    # Checks quadrant and gives coordinates
    coordinate_p1 = get_coordinate(midline_p1, midline_p2, horizontal_moved_p1, vertical_moved_p1)
    # coordinate_p1 = (int(midline_p1[0] - horizontal_moved_p1), int(midline_p1[1] + vertical_moved_p1))
    # print("P1 COORDINATES:", coordinate_p1)
    # Draw new coordinates
    cv2.circle(img, coordinate_p1, 3, (0,0,0), -1)
    cv2.imshow('image', img)
    
    # From the ending point, I want 10m
    # Get ratio
    ratio_p2 = p2 / line_distance
    # Get pixels moved
    horizontal_moved_p2 = ratio_p2 * horizontal_diff
    vertical_moved_p2 = ratio_p2 * vertical_diff
    # print(horizontal_moved_p2)
    # print(vertical_moved_p2)
    # Get new coordinates
    # Checks quadrant and gives coordinates
    coordinate_p2 = get_coordinate(midline_p2, midline_p1, horizontal_moved_p2, vertical_moved_p2)
    # coordinate_p2 = (int(midline_p2[0] + horizontal_moved_p2), int(midline_p2[1] - vertical_moved_p2))
    # print("P2 COORDINATES:", coordinate_p2)
    # Draw new coordinates
    cv2.circle(img, coordinate_p2, 3, (0,0,0), -1)
    cv2.imshow('image', img)

    coordinate = list(coordinate_p1)
    total_pixels_start = p1

    while True:
        # Check which quadrant
        # Get the distance of the new coordinates
        length_new = pytha(coordinate, coordinate_p1)
        # print("\nNew length:", length_new)

        # Normalize
        alpha = length_new / line_distance

        # Put in formula for number of pixels
        number_pixel = formula(alpha) * line_distance
        # print("Number of Pixels:", number_pixel)

        # Total pixels from starting midline for ratio of triangle
        total_pixels_start += number_pixel
        # print("Total Pixels:", total_pixels_start)

        # Get ratio
        ratio = total_pixels_start / line_distance

        # If you cross midline, dont add dot.
        if total_pixels_start > z:
            break

        # Check coordinates moved
        horizontal_moved = ratio * horizontal_diff
        vertical_moved = ratio * vertical_diff
        # print("Hori/Verti Moved:", horizontal_moved, vertical_moved)

        # Get new coordinates
        coordinate = get_coordinate(midline_p1, midline_p2, horizontal_moved, vertical_moved)
        # coordinate = (int(midline_p1[0] + horizontal_moved), int(midline_p1[1] + vertical_moved))

        # If coordinate crosses the ending point, break it
        if not check_coordinate(midline_p1, midline_p2, coordinate_p2, coordinate):
            break

        dots_midline.append((coordinate, number_pixel))
        # f.write(str(number_pixel) + " ")
        
        print("Midline Coordinates:", coordinate)
        # Draw new coordinates
        cv2.circle(img, coordinate, 3, (0,0,255), -1)
        cv2.imshow('image', img)

    print("")
    # f.write("\n")
    for i in dots_midline:
        cv2.circle(img, tuple(i[0]), 3, (0,0,255), -1)
        cv2.imshow('image', img)
    
    
    # NEW TESTING
    f.write(str(coordinate_p1[0]) + " " + str(coordinate_p1[1]) + "\n")
    f.write(str(p1) + "\n")
    f.write(str(p2) + "\n")
    f.write(str(line_distance) + "\n")
    f.write(str(midline_eq[0]) + " " + str(midline_eq[1]))
    f.close()

    # points = [[960, 427], [1027, 422], [1108, 395], [1124, 454], [1110, 514], [1110, 564], [1095, 671], [1136, 642], [1027, 657], [965, 659], [850, 656], [898, 658], [865, 570], [879, 512], [847, 440], [888, 411]]
    # pixels = []
    # for point in points:
    #     f.write(str(point[0]) + " " + str(point[1]) + " ")

    #     length_new = pytha(point, coordinate_p1)
    #     # print("\nNew length:", length_new)

    #     # Normalize
    #     alpha = length_new / line_distance

    #     # Put in formula for number of pixels
    #     number_pixel = formula(alpha) * line_distance

    #     pixels.append(number_pixel)

    # f.write("\n")
    # for pixel in pixels:
    #     f.write(str(pixel) + " ")
    # f.close()

# Make buckets. Saves the list with list of tuple of tuples of coordinates [((x1,y1), (x2,y2)), ((x,y)..]
def make_buckets(img):
    global midline_eq, edge_1_eq, edge_2_eq, dots_midline, f # Tuples of gradient, y-intercept
    for i in range(0, len(dots_midline)-1):
        # Get dot coordinates of first dot
        dot_1_coordinates = dots_midline[i][0]
        # Get perpendicular line
        perp_line_1 = perpendicular(dot_1_coordinates, midline_eq)
        print("\nPerpendicular Line:", perp_line_1)
        # Find intersection of perpendicular line 1 with 2 edges and get 2 coordinates
        # Intersection Edge 1
        intersection_cord_dot_1_eq_1 = intersection(perp_line_1, edge_1_eq)
        # Intersection Edge 2
        intersection_cord_dot_1_eq_2 = intersection(perp_line_1, edge_2_eq)
        
        # Get dot coordinates of second dot
        dot_2_coordinates = dots_midline[i+1][0]
        # Get perpendicular line
        perp_line_2 = perpendicular(dot_2_coordinates, midline_eq)
        # Find intersection of perpendicular line 1 with 2 edges and get 2 coordinates
        # Intersection Edge 1
        intersection_cord_dot_2_eq_1 = intersection(perp_line_2, edge_1_eq)
        # Intersection Edge 2
        intersection_cord_dot_2_eq_2 = intersection(perp_line_2, edge_2_eq)
        x1=str(intersection_cord_dot_1_eq_1[0])
        y1=str(intersection_cord_dot_1_eq_1[1])
        x2=str(intersection_cord_dot_1_eq_2[0])
        y2=str(intersection_cord_dot_1_eq_2[1])
        x3=str(intersection_cord_dot_2_eq_1[0])
        y3=str(intersection_cord_dot_2_eq_1[1])
        x4=str(intersection_cord_dot_2_eq_2[0])
        y4=str(intersection_cord_dot_2_eq_2[1])
        line=x1+' ' +y1+' '+x2+' '+y2+' '+x3+' '+y3+' '+x4+' '+y4+' '
        # print(line)
        # f.write(line)
        print("\nINTERSECTIONS")
        print(intersection_cord_dot_1_eq_1, intersection_cord_dot_1_eq_2)
        print(intersection_cord_dot_2_eq_1, intersection_cord_dot_2_eq_2)

        buckets.append(((intersection_cord_dot_1_eq_1, intersection_cord_dot_1_eq_2), (intersection_cord_dot_2_eq_1, intersection_cord_dot_2_eq_2)))

        cv2.line(img, intersection_cord_dot_1_eq_1, intersection_cord_dot_1_eq_2, (0,0,0), 3)
        cv2.imshow('image', img)
        cv2.line(img, intersection_cord_dot_2_eq_1, intersection_cord_dot_2_eq_2, (0,0,0), 3)
        cv2.imshow('image', img)
    print(buckets)


# HELPER FUNCTIONS

# Gradient
def gradient(p1, p2):
    if p2[0] - p1[0] == 0:
        return 999999999999999999999999999999
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

# Y-Intercept
def y_intercept(p, gradient):
    return p[1] - (gradient * p[0])

# Perpendicular line
def perpendicular(p, equation):
    if equation[0] != 0:
        gradient = -1 / equation[0] # The inverse gradient of the midline
    else:
        gradient = 999999999999999999999999999999
    yintercept = y_intercept(p, gradient)
    return (gradient, yintercept)

# Finds intersection points of two equations
def intersection(equation1, equation2):
    intersection_X = abs((equation2[1] - equation1[1]) / (equation1[0] - equation2[0]))
    intersection_Y = abs((equation1[0] * intersection_X) + equation1[1])
    print("Inside Intersection:", intersection_X, intersection_Y)
    print("")
    return (int(intersection_X), int(intersection_Y))

# Pythagoras
def pytha(point1, point2):
    diff_y = (point1[0] - point2[0])**2
    diff_x = (point1[1] - point2[1])**2
    return (diff_x + diff_y)**0.5

# Gets coordinate based on midline quadrant positions
def get_coordinate(starting_point, ending_point, horizontal_moved, vertical_moved):
    # When X starting is greater than X ending
    if starting_point[0] > ending_point[0]:
        # When Y starting is greater than Y ending
        if starting_point[1] > ending_point[1]:
            # Decrease X and Decrease Y
            return (int(starting_point[0] - horizontal_moved), int(starting_point[1] - vertical_moved))
        # When Y starting is smaller than Y ending
        else:
            # Decrease X and Increase Y
            return (int(starting_point[0] - horizontal_moved), int(starting_point[1] + vertical_moved))
    # When X starting is smaller than X ending
    else:
        # When Y starting is greater than Y ending
        if starting_point[1] > ending_point[1]:
            # Increase X and Decrease Y
            return (int(starting_point[0] + horizontal_moved), int(starting_point[1] - vertical_moved))
        else:
            # Increase X and Increase Y
            return (int(starting_point[0] + horizontal_moved), int(starting_point[1] + vertical_moved))

# Checks if coordinate has passed its end point
def check_coordinate(starting_point, ending_point, coordinate_p2, coordinate):
    # When X starting is greater than X ending
    if starting_point[0] > ending_point[0]:
        # When Y starting is greater than Y ending
        if starting_point[1] > ending_point[1]:
            if (coordinate[1] < coordinate_p2[1]) or (coordinate[0] < coordinate_p2[0]):
                return False
        # When Y starting is smaller than Y ending
        else:
            if (coordinate[1] > coordinate_p2[1]) or (coordinate[0] < coordinate_p2[0]):
                return False
    # When X starting is smaller than X ending
    else:
        # When Y starting is greater than Y ending
        if starting_point[1] > ending_point[1]:
            if (coordinate[1] < coordinate_p2[1]) or (coordinate[0] > coordinate_p2[0]):
                return False
        else:
            if (coordinate[1] > coordinate_p2[1]) or (coordinate[0] > coordinate_p2[0]):
                return False
    return True

# Formula from sir
def formula(alpha):
    return ((1 - alpha) * (p1 / z)) + (alpha * (p2 / z))


def main():
    print("")
    # reading the image 
    img = cv2.imread('image.jpg', 1) 
  
    # displaying the image 
    cv2.imshow('image', img)
  
    # setting mouse hadler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event)
  
    # # # wait for a key to be pressed to exit 
    cv2.waitKey(0)

    # # # close the window 
    cv2.destroyAllWindows()

# driver function 
if __name__=="__main__":
    main()
