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

function detectFaces()

    images = loadImages()
    num_images = length(images)
    turtles_labels = collect(1:num_images)

    # Preprocess the images
    img_size = (300, 300)
    imgs = [imresize(load(img), img_size) for img in images]

    train_images = imgs[1:round(Int, num_images*0.8)]
    train_labels = turtles_labels[1:round(Int, num_images*0.8)]
    test_images = imgs[round(Int, num_images*0.8) + 1:num_images]
    test_labels = turtles_labels[round(Int, num_images*0.8)+1:num_images]

    #preprocess images and make a prediction
    #better to extract turtles features?
    model = Chain(
        Conv((3, 3), 1 => 32, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu),
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(64 * 7 * 7, 400),
        softmax)

    loss(x, y) = crossentropy(model(x), y)
    opt = Flux.setup(Adam(), model)
    Flux.train!(loss, train_images, train_labels, opt)

    accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
    acc = accuracy(test_images, test_labels)



    turtle_counts = Dict{Int, Int}() # Initialize an empty dictionary

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
    end


end

end
