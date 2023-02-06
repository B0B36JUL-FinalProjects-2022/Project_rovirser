using Flux
using Statistics
using Plots

# Write your package code here.
export detectFaces, loss, accuracy

"""
    detecFaces(images)

One paramater of images loaded with a specific shape (30x30), reprocessing them, splitting 
into training and test sets, creating a cnn model and train the samples.

Return: vector of predictions with the test images
"""

function detectFaces()

    images = loadImages()

    # Preprocess the images
    imgs = processImages(images)
    totalImages = toArray(imgs)

    # We split the data to get train and test sets
    X_train, y_train, X_test, y_test = loadData(totalImages)

    loss(X_train, y_train)

    model = Chain(
        Conv((2,2), 1=>16, relu),
        MaxPool((2,2)),
        Conv((2,2), 16=>8, relu),
        MaxPool((2,2)),
        flatten,
        Dense(288, size(y_train,1)),
        softmax,
    )

    return model

end

function loss(X, y)
    loss = -1 * sum(y .* log.(X))
    return loss
end

function accuracy(X, y)
    return mean(onecold(model(x)) .== onecold(y))
end

