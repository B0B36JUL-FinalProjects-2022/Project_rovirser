using Flux
using Flux: params
using Flux: flatten
using Statistics
using Plots
using Flux.Data: DataLoader


# Write your package code here.
export detectFaces, loss, accuracy, Adam, calculate_gradient, evaluation

"""
    detecFaces()

One paramater of images loaded with a specific shape (100x100), reprocessing them, splitting 
into training and test sets, creating a cnn model and train the samples.

Return: vector of predictions with the test images
"""

sigmoid(x) = 1 / (1 .+ exp.(-x))
der_sigmoid(x) = sigmoid(x) * (1 .- σ(x))

function train_sgd(X_train, y_train, num_epochs, learning_rate)
    # Initialize the weights randomly
    W = randn(400, 900)

    # Loop over the training set for the specified number of epochs
    for epoch = 1:num_epochs
        # Shuffle the training set
        shuffle!(X_train)

        # Loop over each image in the training set
        for i in 1:size(X_train, 4)
            # Flatten the image into a 900-dimensional vector
            x = reshape(X_train[:, :, 1, i], 900)

            # Compute the predicted output
            y_pred = sigmoid.(W * x)

            # Compute the error
            y_true = y_train[:, i]
            error = y_pred - y_true

            # Update the weights
            W -= learning_rate * error * x' * der_sigmoid(W * x)
        end
    end

    #Return the learned weights
    return W
end

function predict(X, W)
    # Make predictions for each image in the test set
    y_pred = sigmoid.(W * reshape(X, 900))

    # Return the index of the maximum prediction
    return argmax(y_pred)
end

function compute_accuracy(X, y, W)
    # Make predictions for each image in the test set
    y_pred = [predict(X[:, :, 1, i], W) for i in 1:size(X, 4)]

    # Compute the accuracy
    accuracy = sum(y_pred .== argmax(y, dims=1)') / length(y_pred)

    # Return the accuracy
    return accuracy
end

function count_turtles(y)
    # Count the number of times each turtle appears in the data
    turtle_counts = sum(y, dims=2)'

    # Return the turtle counts
    return turtle_counts
end

function detectFaces()

    images = loadImages()

    # Preprocess the images
    imgs = processImages(images)
    totalImages = toArray(imgs)

    # We split the data to get train and test sets
    X_train, y_train, X_test, y_test = loadData(totalImages)

    return X_train, y_train, X_test, y_test

end

function count_turtles(y)
    # Count the number of times each turtle appears in the data
    turtle_counts = sum(y, dims=2)'

    # Return the turtle counts
    return turtle_counts
end

X_train, y_train, X_test, y_test = detectFaces()

# Train the model
W = train_sgd(X_train, y_train, 10, 0.01)

# Compute the accuracy on the test set
accuracy = compute_accuracy(X_test, y_test, W)
println("Accuracy: $accuracy")

# Count the number of times each turtle appears in the data
turtle_counts = count_turtles(y_train)
println("Turtle counts: $turtle_counts")


