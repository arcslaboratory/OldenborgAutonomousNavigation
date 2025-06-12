import time

from jetbot import Camera
from PIL import Image

camera = Camera()
time.sleep(5) # wait 5 seconds before first image to give time to put camera in right position
for i in range(500): #500 images
    image_arr = camera.value
    print(i)
    image = Image.fromarray(image_arr)
    image.save("all_images/" + str(i)+".png")
    time.sleep(1)
