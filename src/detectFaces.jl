using Flux
using Flux: params
using Flux: crossentropy
using Flux.Data: DataLoader
using Statistics
using Random
using DataFrames
using DecisionTree
# Write your package code here.
export detectFaces, train_model!, decisionTree

function decisionTree()

    images = loadImages()
    size(images)

    num_images = length(images)
    num_turtles = 400
    turtles = collect(1:num_turtles)

    # Preprocess the images
    imgs = processImages(images)

    #create a matrix where each row will be an image
    x = zeros(num_images, 300*300)
    for i in 1:num_images
        image_vector = vec(imgs[i])
        x[i, :] = image_vector
    end

    #train images and different classes
    train_images = x[1:round(Int, num_images*0.8), :]
    labels = collect(1:400)

    #model and prediction
    model = build_forest(labels, train_images, 20, 50, 1.0)
    predTest = apply_forest(model, x)

    return predTest

end


function detectFaces(max_epochs = 1000)

    images = loadImages()
    size(images)

    num_images = length(images)
    num_turtles = 400
    turtles = collect(1:num_turtles)

    # Preprocess the images
    imgs = processImages(images)
    totalImages = toArray(imgs)

    X_train, y_train, X_test, y_test = loadData(totalImages, true)
    labels =  

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
    loss(X, y) = crossentropy(model(X_train), y)
    train_model!(model, loss, X_train, y_train)

    #Check the accuracy
    accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
    "Test accuracy = " * string(accuracy(X_test, y_test))

    predictions = onecold(model(test_images))


end

function train_model!(m, L, X, y; batchsize = 128, n_epochs = 10)
    batches = DataLoader((X, y), batchsize = batchsize, shuffle = true)
    opt = Flux.Adam()
    for _ in 1:n_epochs
        for (x, y) in batches
            Flux.train!(L, params(m), [(x, y)], opt)
        end
    end
end



