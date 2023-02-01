using Revise
import SeaTurtleID as turtle

#loading images with reshaped to 30x30
images = loadImages()

display(images[1])

#Converting  images to float and to Gray
processed_images = processImages(images)

display(processed_images[1])

#labels vectors of the turtles
labels = loadLabels()

