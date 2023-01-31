using Flux
using Images
using ColorTypes
using Statistics

# Write your package code here.
export detectFaces

function detectFaces(max_epochs = 1000)

    images = loadImages()
    size(images)

    num_images = length(images)
    num_turtles = 400
    turtles = collect(1:num_turtles)

    # Preprocess the images
    imgs = processImages(images)

    #Train and test samples
    train_images = imgs[1:round(Int, num_images*0.8)]
    test_images = imgs[round(Int, num_images*0.8) + 1:num_images]
    train_labels = turtles[1:round(Int, num_turtles*0.8)]
    test_labels = turtles[round(Int, num_turtles*0.8)+1:num_turtles]
    

    # Define a CNN using Flux
    model = Chain(
        Conv((3, 3), 1 => 16, pad=(1, 1), relu),
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, pad=(1, 1), relu),
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(32 * 7 * 7, 10),
        softmax
      )
    

    # Train the model
    loss(x, y) = Flux.mse(model(x), y)
    optimizer = Flux.setup(Adam(), model)


   # Train for a specified number of epochs
    for epoch in 1:max_epochs
        for (x, y) in zip(train_images, train_labels)
            gs = gradient(() -> loss(x, y), Flux.params(model))
            optimizer.(gs)
        end
        println("Epoch: $epoch, Loss: $(mean(loss.(train_images, train_labels)))")
    end


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