using Revise
import SeaTurtleID as turtle

loadDataset()
images = loadImages()
processImages(images)
detectFaces(max_pochs = 1000)

