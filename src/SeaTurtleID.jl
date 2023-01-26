module SeaTurtleID

using  JSON, DataFrames, Images, Flux
import FLux: params
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
    turtlesID = collect(1:400)

    num_images = length(images)
    train_images = images[1:round(Int, num_images*0.8)]
    train_labels = turtlesID[1:round(Int, 400*0.8)]
    test_images = images[round(Int, num_images*0.8) + 1:num_images]
    test_labels = turtlesID[round(Int, 400*0.8)+1:400]

    # Define the CNN architecture
    model = Chain(
        Conv((3, 3), 3 => 32, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu),
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(64 * 7 * 7, 400), # number of turtles in the dataset
        softmax)

    # Define the loss function and the optimizer
    loss(x, y) = crossentropy(model(x), y)
    opt = ADAM(params(model))

    # Train the model
    Flux.train!(loss, train_images, train_labels, opt)

    # Test the CNN
    accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
    acc = accuracy(test_imgs, test_labels)
    
    predictions = onecold(model(test_images))

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
