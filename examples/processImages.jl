using Revise
import SeaTurtleID as turtle

images = loadImages()

for i in 1:10
    display(images[1])
end

processed_images = processImages(images)

typeof(processed_images)