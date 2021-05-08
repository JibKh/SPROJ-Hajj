import numpy as np
import shutil
import cv2
import os
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def check_lengths_frames_text():
    text_cords = open("./text_files/cords.txt", 'r')
    text_cords = text_cords.read().split("\n")

    len_cords_lines = len(text_cords)

    flowframes_dir = os.listdir("/content/flownet2pytorch/output/inference/run.epoch-0-flow-field")
    len_flowframes = len(flowframes_dir)

    frame_dir = os.listdir("/content/frames")
    len_frames = len(frame_dir)

    print(len_cords_lines, len_flowframes, len_frames)

    if len_cords_lines == len_flowframes == len_frames:
        return True

    return False

def read_speed():
    global normalize_meters, coordinate_p1, p1_pixels, p2_pixels, z, midline_eq

    text_speed = open("./text_files/text_speed.txt", 'r')
    text_speed = text_speed.read()

    # Text Speed assigning
    text_speed = text_speed.split("\n")

    normalize_meters = float(text_speed[0])

    coordinate_p1 = text_speed[1].split(" ")
    coordinate_p1[0] = int(coordinate_p1[0])
    coordinate_p1[1] = int(coordinate_p1[1])

    p1_pixels = float(text_speed[2])
    p2_pixels = float(text_speed[3])

    z = float(text_speed[4])

    temp_midline_eq = text_speed[5].split(" ")
    midline_eq = float(temp_midline_eq[0]), float(temp_midline_eq[1])

    # print(normalize_meters)
    # print(coordinate_p1)
    # print(z)

# Read all text files
def read_points():
    global points

    # Read text files
    
    text_cords = open("./text_files/cords.txt", 'r')
    text_cords = text_cords.read()

    text_cords = text_cords.split("\n")

    # Text Coordinates assigning
    points = []
    for line in text_cords:
        points.append(line)

    # text_cords = text_cords.split()
    # for i in range(0, len(text_cords)-1, 2):
    #     points.append([int(text_cords[i]), int(text_cords[i+1])])

def find_points_line(i):
    global points, points_line

    points_line = []

    points_string = points[i]

    if points_string == '':
        return False

    points_string = points_string.split(" ")

    for i in range(0, len(points_string)-1, 2):
        points_line.append([int(points_string[i]), int(points_string[i+1])])

    return True

# Find the number of pixels representing each point
def find_number_pixels():
    global number_pixels, points_line, coordinate_p1, z, midline_eq

    # Here we take the gradient and inverse it. If its 0 then make it infinity
    if midline_eq[0] != 0:
        midline_grad_inv = -1 / midline_eq[0]
    else:
        midline_grad_inv = 99999999999999

    number_pixels = []

    # Loop through all points
    for point in points_line:
        midline_point = find_shortest_midline(point, midline_grad_inv)
        length_new = pytha(midline_point, coordinate_p1)
        
        # length_new = pytha(point, coordinate_p1)

        # Normalize
        alpha = length_new / z

        # Put in formula for number of pixels
        number_pixels.append(formula(alpha) * z)
        
# Find speed of each point
def find_speed(flow):
    global points_line, number_pixels, speed, velocity

    speed = []
    velocity = []

    # Read flow
    flo = read_flow(flow)
    flow = make_flow(flo) # Contains numpy array of velocity values.
    flo.close()
    flow = np.transpose(flow, (1, 0, 2))

    for i in range(0, len(points_line)):
        velocity.append(flow[points_line[i][0]][points_line[i][1]])
        value = find_speed_single(flow[points_line[i][0]][points_line[i][1]], number_pixels[i])
        speed.append(value)

    # print("\nAll Speed:", speed)

# Find resultant speed of a vector
def find_speed_single(velocity, number_pixel):
    global normalize_meters
    final_velocity = ((velocity[0] ** 2) + (velocity[1] ** 2)) ** 0.5
    # final_velocity = 22

    frame_second = 30

    return((final_velocity*frame_second / number_pixel) * normalize_meters)

# Clustering
def cluster():
    global speed, velocity, points_line

    clusters = 4

    velocity = np.array(velocity)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(velocity)

    kmeans = KMeans(
        init="random",
        n_clusters=clusters,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    kmeans.fit(scaled_features)

    kmeans.inertia_

    kmeans.cluster_centers_

    kmeans.n_iter_

    # print("\nLabels:", kmeans.labels_)

    # # Divide the labeled points and speed
    # clustered_points = dict()
    # clustered_speed = dict()
    # clustered_velocity = dict()

    # for i, label in enumerate(kmeans.labels_):
    #     try:
    #         clustered_points[str(label)].append(points_line[i])
    #         clustered_speed[str(label)].append(speed[i])
    #         clustered_velocity[str(label)].append(velocity[i])
    #     except:
    #         clustered_points.update({str(label):[points_line[i]]})
    #         clustered_speed.update({str(label):[speed[i]]})
    #         clustered_velocity.update({str(label):[velocity[i]]})

    # print(clustered_points)
    # print(clustered_speed)

    # # Find average for each label
    # for label in clustered_speed:
    #     print("Average of ", label, "is:", average(clustered_speed[label], len(points_line)))

    # return clustered_points, clustered_velocity

    # for i in range(0, len(velocity)):
    #     x = points[i][0]
    #     y = points[i][1]
        
    #     new_cord = (int(x + velocity[i][0]), int(y + velocity[i][1]))

    #     cv2.arrowedLine(img, (x,y), new_cord, (255,0,0), 3)
    #     cv2.imshow('image', img)

    # velocity_labels = [[], [], [], [], [], [], [], []] # Starts from top and goes clockwise. Contains the points.
    # points_labels = [[], [], [], [], [], [], [], []] # Starts from top and goes clockwise. Contains the points.
    # speed_labels = [[], [], [], [], [], [], [], []] # Starts from top and goes clockwise. Contains the points.
    velocity_labels = [[], [], [], [], []] # Starts from top and goes clockwise. Contains the points.
    points_labels = [[], [], [], [], []] # Starts from top and goes clockwise. Contains the points.
    speed_labels = [[], [], [], [], []] # Starts from top and goes clockwise. Contains the points.
    average_speed = []

    for i, val in enumerate(kmeans.labels_):
        label = int(val)
        velocity_labels[label-1].append(velocity[i])
        points_labels[label-1].append(points_line[i])
        speed_labels[label-1].append(speed[i])

        for i in range(0, len(speed_labels)):
            average_speed.append((average(speed_labels[i], len(points_line)), len(speed_labels[i])))
            # print("Average of ", i, "is:", average(speed_labels[i], len(points_line)), "for", len(speed_labels[i]), "people.")

    return points_labels, velocity_labels, speed_labels, average_speed

def cluster_2():
    global velocity, speed, points_line, total_total

    velocity_labels = [[], [], [], [], [], [], [], [], []] # Starts from top and goes clockwise. Contains the points. Last is less than X speed.
    points_labels = [[], [], [], [], [], [], [], [], []] # Starts from top and goes clockwise. Contains the points. Last is less than X speed.
    speed_labels = [[], [], [], [], [], [], [], [], []] # Starts from top and goes clockwise. Contains the points. Last is less than X speed.
    average_speed = [] # Average speed of each cluster with its number of people
    total_average_speed = 0 # The avg speed of the whole crowd

    for i, vel in enumerate(velocity):
        # print("\nVelocity:", vel)
        quadrant = check_quadrant(vel)
        # print("Quadrant:", quadrant)
        line_eq = make_line_eq(vel)
        # print("LineEq:", line_eq)
        degree = find_degree(line_eq[0])
        # print("Degree:", degree)
        # print("Speed:", speed[i])
        label = check_label(quadrant, degree, speed[i]) # Check quadrant and degree for position. Check speed if less than X.
        # print("Final Label:", label)

        velocity_labels[label-1].append(velocity[i])
        points_labels[label-1].append(points_line[i])
        speed_labels[label-1].append(speed[i])
    
    for i in range(0, len(speed_labels)):
        average_speed.append((average(speed_labels[i], len(speed_labels[i])), len(speed_labels[i])))
        total_average_speed += average_speed[i][0]
        # print("Average of ", i, "is:", average_speed[i][0], "for", average_speed[i][1], "people.")

    # total_average_speed = total_average_speed / len(points_line)
    total_average_speed = total_average_speed
    total_total += total_average_speed

    return points_labels, velocity_labels, speed_labels, average_speed, total_average_speed
    
def output_arrows(i, frame_name, clustered_points, clustered_velocity, clustered_speed, average_speed, total_average_speed):
    global points_line

    # Draw arrows
    img = cv2.imread(frame_name)

    # if i == 0:
    #     # img = cv2.imread("./frames/000000.jpg")
    #     img = cv2.imread("image.jpg")
    # else:
    #     img = cv2.imread("./frames/000100.jpg")

    colors = [(90,90,90), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255), (100,100,140), (255,0,0)]

    for i in range(0, len(clustered_points)):
        for j in range(0, len(clustered_points[i])):
            x = clustered_points[i][j][0]
            y = clustered_points[i][j][1]

            new_cord = (int(x + clustered_velocity[i][j][0]), int(y + clustered_velocity[i][j][1])) # Get new coordinate

            # Multiple length of coordinate with speed to get length of arrow
            distance = pytha(new_cord, (x, y))
            # print("\nDistance:", distance)

            if distance != 0:

                greater_distance = distance * clustered_speed[i][j] * 2
                # print("Greater Distance:", greater_distance)

                # Get new coordinates after speed multiplication
                ratio = greater_distance / distance
                # ratio = ratio / 
                # print("Ratio:", ratio)
                
                x_final = x + int((new_cord[0] - x) * ratio)
                y_final = y + int((new_cord[1] - y) * ratio)

                new_cord = (x_final, y_final)

            color = colors[int(i)]
            # print("color:", color)
            cv2.arrowedLine(img, (x,y), new_cord, color, 3)

    # for i in clustered_points:
    #     print(i)
    #     for j in range(0, len(clustered_points[str(i)])):
    #         x = clustered_points[i][j][0]
    #         y = clustered_points[i][j][1]

    #         new_cord = (int(x + clustered_velocity[i][j][0]-20), int(y + clustered_velocity[i][j][1]-20))

    #         color = colors[int(i)]
    #         print("color:", color)
    #         cv2.arrowedLine(img, (x,y), new_cord, color, 2)

    # Add border
    img = cv2.copyMakeBorder(img, 180, 0, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    arrow_directions_text = ["T", "TR", "R", "BR", "B", "BL", "L", "TL", "S"]
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX 
    for i, color in enumerate(colors):
            factor_arrow = int((i + 2.5) * 115) - 205
            factor_speed = int((i + 2) * 115) - 195
            factor_count = int((i + 2) * 115) - 195
            
            cv2.putText(img, arrow_directions_text[i], (factor_arrow-8, 30), font, 1, color, 2)
            cv2.arrowedLine(img, (factor_arrow, 100), (factor_arrow, 50), color, 6, tipLength = 0.3)
            cv2.putText(img, str(round(average_speed[i][0], 1)) + " m/s", (factor_speed, 130), font, 0.7, color, 2)
            cv2.putText(img, str(average_speed[i][1]) + " ppl", (factor_count, 160), font, 0.7, color, 2)

    cv2.putText(img, "Avg Speed: " + str(round(total_average_speed, 2)), (1080, 100), font, 0.7, (255,255,255), 2)
    cv2.putText(img, "Total Ppl:  " + str(len(points_line)), (1080, 150), font, 0.7, (255,255,255), 2)

    # Write image
    cv2.imwrite("./output_frames/" + frame_name.split("/")[-1], img)

    # Display image
    # cv2.imshow('image', img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Helper Functions

def find_shortest_midline(point, midline_grad_inv):
    global midline_eq

    midline_y_inv = point[0]-midline_grad_inv

    x = (midline_y_inv - midline_eq[1]) / (midline_eq[0] - midline_grad_inv)
    y = (x * midline_eq[0]) + midline_eq[1]

    return (x, y)


def check_quadrant(vel):
    if vel[0] > 0: # Positive X
        if vel[1] > 0: # Positive Y
            return 3
        else:
            return 1
    else: # Negative X
        if vel[1] > 0: # Positive Y
            return 4
        else:
            return 2

def make_line_eq(vel):
    gradient = vel[1] / vel[0]
    y_intercept = vel[1] - (vel[0] * gradient)
    
    return (gradient, y_intercept)

def find_degree(gradient):
    return abs(math.degrees(math.atan(gradient)))

def check_label(quadrant, degree, speed):
    if speed <= 0.1:
        return 9

    if quadrant == 1:
        if degree <= 22.5:
            return 3
        elif degree <= 67.5:
            return 2
        elif degree <= 90:
            return 1
        else:
            print("Error in degree quadrant 1")
    elif quadrant == 2:
        if degree <= 22.5:
            return 7
        elif degree <= 67.5:
            return 8
        elif degree <= 90:
            return 1
        else:
            print("Error in degree quadrant 2")
    elif quadrant == 3:
        if degree <= 22.5:
            return 3
        elif degree <= 67.5:
            return 4
        elif degree <= 90:
            return 5
        else:
            print("Error in degree quadrant 3")
    elif quadrant == 4:
        if degree <= 22.5:
            return 7
        elif degree <= 67.5:
            return 6
        elif degree <= 90:
            return 5
        else:
            print("Error in degree quadrant 4")
    else:
        print("No quadrant found")

# Formula from sir
def formula(alpha):
    global p1_pixels, p2_pixels, z
    return ((1 - alpha) * (p1_pixels / z)) + (alpha * (p2_pixels / z))

# Pythagoras
def pytha(point1, point2):
    diff_y = (point1[0] - point2[0])**2
    diff_x = (point1[1] - point2[1])**2
    return (diff_x + diff_y)**0.5

# From flownet code
def make_flow(flo):
    tag = np.fromfile(flo, np.float32, count=1)[0]
    width = np.fromfile(flo, np.int32, count=1)[0]
    height = np.fromfile(flo, np.int32, count=1)[0]
    nbands = 2
    tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
    flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    return flow

def read_flow(flow):
    return open("/content/flownet2pytorch/output/inference/run.epoch-0-flow-field/"+flow, "r") 

def average(values, length):
    # print("Values:", values)
    # print("Length:", length)
    num = sum(values)
    if length == 0:
        return 0
    else:
        return num / length

# Copy frame
def copy_frame(frame):
    shutil.copyfile(frame, "./output_frames/" + frame.split("/")[-1])

def main():
    global normalize_meters, number_pixels, coordinate_p1, p1_pixels, p2_pixels, z, points, speed, velocity, points_line, midline_eq, total_total

    # Read all speed values. 
    read_speed() # normalize_meters, coordinate_p1, p1_pixels, p2_pixels, z, midline_eq
    read_points() # Read points. points variable is list of the lines in a frame
    
#     if not check_lengths_frames_text():
#         print("Lengths of frames, flowframes and coordinate text file does not match")
    
    for i, flow in enumerate(sorted(os.listdir("/content/flownet2pytorch/output/inference/run.epoch-0-flow-field/"))):
        frame_name = "/content/frames/" + flow[0:-4] + ".png"
        
        # Checks if a line is empty. If it is then Copy that frame image and continue
        if not find_points_line(i): # Get the line points. The variable "points_line" now contains a list of only that frames points.
            copy_frame(frame_name)
            continue

        find_number_pixels() # Get the pixels of each point in the line
        find_speed(flow) # Find speed. For each point, the speed is stores in a list like speed = [1,2,3]

        points_labels, velocity_labels, speed_labels, average_speed, total_average_speed = cluster_2()
        output_arrows(i, frame_name, points_labels, velocity_labels, speed_labels, average_speed, total_average_speed)
        # clustered_points, clustered_velocity = cluster() # Cluster and output average speed
        # output_arrows(i, clustered_points, clustered_velocity)


    cv2.waitKey(0)

# driver function 
if __name__=="__main__":
    main()
