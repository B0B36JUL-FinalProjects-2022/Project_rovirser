module SeaTurtleID

using JSON, DataFrames
using Flux
using Images


# Write your package code here.
export loadDataset, loadImages


function loadDataset()

    # Define the path to the JSON file
    file_path = "./data/annotations.json"

    # Read the JSON file
    data = JSON.parsefile(file_path)

    # Convert the JSON data to a dataset
    column_data = data["images"]
    dataset = DataFrame(column_data)

    return dataset

end

function loadImages()

    dataset = loadDataset()
    images = dataset[:, "path"]

    # Add ./data/ to access the correct path
    for i in 1:7582
        images[i] = "./data/"*images[i]
    end

    return images
end

function processImages(images :: Array{String})
    img_size = (600, 600)
    imgs = [imresize(load(img), img_size) for img in images]
    return imgs
end

function detectFaces()

    images = loadImages()
    num_images = length(images)
    turtles_labels = collect(1:num_images)

    # Preprocess the images
    imgs = processImages(images)

    #grayscale?

    train_images = imgs[1:round(Int, num_images*0.8)]
    train_labels = turtles_labels[1:round(Int, num_images*0.8)]
    test_images = imgs[round(Int, num_images*0.8) + 1:num_images]
    test_labels = turtles_labels[round(Int, num_images*0.8)+1:num_images]
   
    # resnet model?



    #= turtle_counts = Dict{Int, Int}() # Initialize an empty dictionary

    # Iterate through the predictions
    for turtle in predictions
        if haskey(turtle_counts, turtle) # Check if turtle already in dictionary
            turtle_counts[turtle] += 1 # If so, increment count
        else
            turtle_counts[turtle] = 1 # If not, add turtle to dictionary with count 1
        end
    end

    # Print out turtle counts
    for turtle in keys(turtle_counts)
        println("Turtle $(turtle) appears $(turtle_counts[turtle]) times")
    end =#


end

end
