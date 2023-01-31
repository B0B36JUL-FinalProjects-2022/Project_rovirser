using Revise
import SeaTurtleID: sea

loadDataset()
images = loadImages()
processImages(images :: Array{String})
detectFaces(max_pochs = 1000)