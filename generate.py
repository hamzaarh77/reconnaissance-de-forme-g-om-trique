import cv2
import os
import numpy as np
import random

output= ["data", "data/triangle", "data/rectangle", "data/cercle"]
for e in output:
    if not os.path.exists(e):
        print("not found")
        os.makedirs(e)

# declaration constante 
IMG_SIZE = 200

def draw_rectangle(image):
    pt1 = (random.randint(20,100), random.randint(20,100))
    pt2 = (random.randint(pt1[0]+20, 180), random.randint(pt1[1]+20, 180))
    color = (255,255,255)
    thickness = random.randint(1,3) # epaisseur aleatoire 
    cv2.rectangle(image, pt1, pt2, color, thickness)

 

def draw_triangle(image):
    pt1 = (random.randint(50, 150), random.randint(50, 150))
    pt2 = (random.randint(50, 150), random.randint(50, 150))
    pt3 = (random.randint(50, 150), random.randint(50, 150))

    pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
    thickness = random.randint(1,3)
    color = (255, 255, 255)
    cv2.polylines(image, [pts],isClosed= True, color= color, thickness = thickness)

def draw_circle(image):
    center = (random.randint(50, 150), random.randint(50,150))
    radius = random.randint(20, 50)
    color =(255, 255, 255)
    thickness = random.randint(1,3)
    cv2.circle(image, center, radius, color, thickness)



def generate_image():
    image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype = np.uint8)

    forme = random.choice(["cercle", "triangle", "rectangle"])

    if forme == "cercle":
        draw_circle(image)
    elif forme == "rectangle":
        draw_rectangle(image)
    else:
        draw_triangle(image)
    
    return image, forme 


num_images = 200
for i in range(num_images):
    image, forme = generate_image()
    filename = f"data/{forme}/image_{i}.png"
    cv2.imwrite(filename, image)



