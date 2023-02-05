using Flux
using Flux: params
using Flux: crossentropy
using Flux.Data: DataLoader
using Flux: onecold
using Statistics
using BSON
using Plots

# Write your package code here.
export detectFaces

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
    X_train, y_train, X_test, y_test = loadData(totalImages, true)


end

function loss(X, y)

    loss = -1 * sum(y .* log.(X))
    return loss
end


