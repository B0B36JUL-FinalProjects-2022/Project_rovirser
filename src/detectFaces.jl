using Flux
using Flux: params
using Flux: crossentropy
using Flux.Data: DataLoader
using Flux: onecold
using Statistics
using Random
using DataFrames
using BSON

# Write your package code here.
export detectFaces, train_model!, train_or_load!

function detectFaces()

    images = loadImages()
    size(images)

    num_images = length(images)
    num_turtles = 400
    turtles = collect(1:num_turtles)

    # Preprocess the images
    imgs = processImages(images)
    totalImages = toArray(imgs)

    #display(images[1])

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
    train_model!(m, L, X_train, y_train; n_epochs=30, file_name=file_name)

    #Check the accuracy
    accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))
    "Train accuracy = " * string(accuracy(X_train, y_train)) |> println

    return X_train, X_test, y_train, y_test
end

function train_model!(m, L, X, y;
    opt = Descent(0.1),
    batchsize = 32,
    n_epochs = 10,
    file_name = "")

    batches = DataLoader((X, y); batchsize, shuffle = true)

    for _ in 1:n_epochs
        Flux.train!(L, params(m), batches, opt)
    end

    !isempty(file_name) && BSON.bson(file_name, m=m)

    return
end

function train_or_load!(file_name, m, args...; force=false, kwargs...)

    !isdir(dirname(file_name)) && mkpath(dirname(file_name))

    if force || !isfile(file_name)
        train_model!(m, args...; file_name=file_name, kwargs...)
    else
        m_weights = BSON.load(file_name)[:m]
        Flux.loadparams!(m, params(m_weights))
    end
end


