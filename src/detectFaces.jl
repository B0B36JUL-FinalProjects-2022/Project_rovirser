using Flux
using Flux: params
using Flux: crossentropy
using Flux.Data: DataLoader
using Statistics
using Random

# Write your package code here.
export detectFaces, train_model!

function detectFaces(max_epochs = 1000)

    images = loadImages()
    size(images)

    num_images = length(images)
    num_turtles = 400
    turtles = collect(1:num_turtles)

    # Preprocess the images
    imgs = processImages(images)
    totalImages = toArray(imgs)

    X_train, y_train, X_test, y_test = loadData(totalImages, num_images; onehot=true)

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

function train_model!(m, L, X, y;
    opt = Descent(0.1),
    batchsize = 128,
    n_epochs = 10)

    batches = DataLoader((X, y); batchsize, shuffle = true)

    for _ in 1:n_epochs
        Flux.train!(L, params(m), batches, opt)
    end

    !isempty(file_name) && BSON.bson(file_name, m=m)

    return
end

