using Images
using ColorTypes
using Statistics

export loadImages, processImages

function loadImages()

    dataset = loadDataset()
    images = dataset[:, "path"]
    num = length(images)

    # Add ./data/ to access the correct path
    for i in 1:num
        images[i] = "./data/"*images[i]
    end

    #Load image and resizing to 300x300 each one
    img_size = (300, 300)
    return map(x -> imresize(load(x), img_size), images)

end

function processImages(images)
    
     # Normalize the images between 0 and 1
     images = map(x -> Gray.(x), images)
     images = map(x -> Float32.(x), images)
     images = map(x -> x .- mean(x), images)    #center the data around 0, can help to remove any bias
     images = map(x -> x ./ std(x), images)     #make easier for the model to learn the features in the data

     return images

end
