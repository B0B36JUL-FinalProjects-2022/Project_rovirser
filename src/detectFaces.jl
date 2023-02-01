using Flux
using Flux: params
using Flux: crossentropy
using Flux.Data: DataLoader
using Flux: onecold
using Statistics
using Random
using DataFrames
using BSON
using Plots

# Write your package code here.
export detectFaces, train_model!, train_or_load!


"""
    detecFaces(images)

One paramater of images loaded with a specific shape (30x30), reprocessing them, splitting 
into training and test sets, creating a cnn model and train the samples.

Return: vector of predictions with the test images
"""

function detectFaces(images)

    # Preprocess the images
    imgs = processImages(images)
    totalImages = toArray(imgs)


    X_train, y_train, X_test, y_test = loadData(totalImages, true)

    # Define a CNN using Flux
    Random.seed!(666)
    m = Chain(
        Conv((2,2), 1=>16, relu),
        MaxPool((2,2)),
        Conv((2,2), 16=>8, relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(288, size(y_train,1)),
        softmax,
    )

    # Train the model
    L(X, y) = crossentropy(m(X), y)

    file_name = "model_project.bson"   
    train_model!(m, L, X_train, y_train; n_epochs=20, file_name=file_name)

    #Check the accuracy
    accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
    "Train accuracy = " * string(accuracy(X_train, y_train)) |> println

    predictions = onecold(m(X_test))

    return predictions
end

"""
    train_model!(m, l, X, y)

Training the data with the model and loss function, responsable of dividing the set with batches
Repeats the process epochs times
"""

function train_model!(m, L, X, y;
    opt = Descent(0.1),
    batchsize = 20,
    n_epochs = 10,
    file_name = "")

    batches = DataLoader((X, y); batchsize, shuffle = true)

    for _ in 1:n_epochs
        Flux.train!(L, params(m), batches, opt)
    end

    !isempty(file_name) && BSON.bson(file_name, m=m)

    return
end



