using Images
using ColorTypes
using Statistics

export loadImages, processImages, loadLabels

function loadImages()

    dataset = loadDataset()
    images = dataset[:, "path"]
    num = length(images)

    # Add ./data/ to access the correct path
    for i in 1:num
        images[i] = "./data/"*images[i]
    end

    #Load image and resizing to 300x300 each one
    img_size = (100, 100)
    return map(x -> imresize(load(x), img_size), images)

end

function processImages(images)
    
     # Normalize the images between 0 and 1
     images = map(x -> Gray.(x), images)
     images = map(x -> Float32.(x), images)

     return images

end

function loadLabels()

    dataset = loadDataset()
    path_orig = dataset[:, "path_orig"]
    labels = Array{String}(undef, length(path_orig))

    for i in eachindex(path_orig)
        turtleID = split(path_orig[i], "/")[1]
        labels[i] = turtleID
    end

    return labels
end

