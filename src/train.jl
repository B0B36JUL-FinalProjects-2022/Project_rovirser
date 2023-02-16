using Flux
using Flux: params
using Flux: crossentropy
using Statistics
using Flux.Data: DataLoader
using Random

# Write your package code here.
export loadSets, train, accuracy, sgd

function loadSets()

    images = loadImages()

    # Preprocess the images
    imgs = processImages(images)
    totalImages = toArray(imgs)

    # We split the data to get train and test sets
    X_train, y_train, X_test, y_test = loadData(totalImages)

    return X_train, y_train, X_test, y_test

end

# Define the SGD algorithm for training the CNN model
function sgd(params, gradients, lr)
    for (p, g) in zip(params, gradients)
        p .-= lr .* g
    end
end

# Define the function to compute the accuracy of the model on a given data set
function accuracy(model, X, y)
    predictions = argmax(model(X), dims=1)
    targets = argmax(y, dims=1)
    mean(predictions .== targets)
end

function train(X_train, y_train, epochs, lr, batch_size)

     # Initialize the model
     model = Chain(
        # First convolutional layer
        Conv((5, 5), 1=>16, relu),
        MaxPool((2, 2)),
        # Second convolutional layer
        Conv((5, 5), 16=>32, relu),
        MaxPool((2, 2)),
        # Flatten layer
        x -> reshape(x, :, size(x, 4)),
        # Dense layer
        Dense(512, 400),
        softmax,
      )
    # Define the loss function
    loss(x, y) = crossentropy(model(x), y)

    parameters = Flux.params(model)

    for epoch in 1:epochs
        # Shuffle the training data
        shuffle_idxs = shuffle(1:size(X_train, 4))
        X_train = X_train[:, :, :, shuffle_idxs]
        y_train = y_train[:, shuffle_idxs]

        # Create a data loader from the training data
        data_loader = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)

        # Train on mini-batches
        for (batch_X, batch_y) in data_loader
            # Compute the gradients and update the model parameters
            gs = gradient(() -> loss(batch_X, batch_y), params(model))
            sgd(parameters, gs, lr)
        end

        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
        println("Epoch $epoch: Train accuracy = $train_acc, Test accuracy = $test_acc")
    
    end

    return model

end

