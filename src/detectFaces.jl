using Flux
using Images
using ColorTypes

# Write your package code here.
export detectFaces

function detectFaces(max_pochs = 1000)

    images = loadImages()
    num_images = length(images)
    turtles = collect(1:400)
    num_turtles = 400

    # Preprocess the images
    imgs = processImages(images)

    train_images = imgs[1:round(Int, num_images*0.8)]
    train_labels = turtles[1:round(Int, num_turtles*0.8)]
    test_images = imgs[round(Int, num_images*0.8) + 1:num_images]
    test_labels = turtles[round(Int, num_turtles*0.8)+1:num_turtles] 

    # Define a CNN using Flux
    model = Chain(
        Conv((3, 3), 1 => 32, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu),
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(128, 400),
        identity
    )

    # Move the model to GPU for faster training
    model = gpu(model)
    loss(x, y) = Flux.crossentropy(model(x), y)    
    opt = Flux.setup(Adam(), model)

    for i in 1:max_pochs
        Flux.train!(loss, train_images, train_labels, opt)
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