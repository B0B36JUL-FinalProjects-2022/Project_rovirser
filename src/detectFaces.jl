using Flux
using Flux: params
using Images
using ColorTypes
using Statistics
using Random
using Flux.Data: DataLoader

# Write your package code here.
export detectFaces

function detectFaces(num_turtles = 400; max_epochs = 1000)

    images = loadImages()
    size(images)

    num_images = length(images)
    turtles = collect(1:num_turtles)

    # Preprocess the images
    imgs = processImages(images)
    imgs = toArray(imgs)

    size(imgs)

    X_train, y_train, X_test, y_test = loadData(imgs, turtles, num_turtles; onehot=true)

    # Define a CNN using Flux
    Random.seed!(666)
    model = Chain(
        Conv((2,2), 1=>16, relu),
        MaxPool((2,2)),
        Conv((2,2), 16=>8, relu),
        MaxPool((2,2)),
        flatten,
        Dense(288, size(y_train,1)),
        softmax,
    )
    
    # Train the model
    loss(X, y) = crossentropy(model(X), y)
    
    train_model!(model, loss, X_train, y_train)

    #Check the accuracy
    accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
    "Test accuracy = " * string(accuracy(X_test, y_test))

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

function train_model!(model, loss, X, y;
    opt = Descent(0.1),
    batchsize = 128,
    n_epochs = 10,)

    batches = DataLoader((X, y); batchsize, shuffle = true)

    for _ in 1:n_epochs
        Flux.train!(loss, params(model), batches, opt)
    end

    return
end