using Flux: params
using Flux: flatten
using Statistics
using Plots

# Write your package code here.
export detectFaces, loss, accuracy, Adam, calculate_gradient, train_model

"""
    detecFaces()

One paramater of images loaded with a specific shape (100x100), reprocessing them, splitting 
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

    #Check if this is correct
    model = Chain(
        Conv((2,2), 1=>16, relu),
        MaxPool((2,2)),
        Conv((2,2), 16=>8, relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(288, size(y_train,1)),
        softmax,
    )

    return model

end

function loss(X, y)
    loss = -sum(log.(X) .* y)
    return loss
end

function accuracy(X, y, classes)
    return mean(onecold(model(X), classes) .== onecold(y, classes))
end

function Adam(model; learning_rate = 0.001, epochs=10, decay_rate1=0.9, decay_rate2=0.999, ε=1e-8)

    params = params(model)
    m = [zeros(size(p)) for p in params]        # moving average
    v = [zeros(size(p)) for p in params]        # gradient average

    for _ in 1:epochs

        g = calculate_gradient(() -> loss(model(X), y), params)

    end

end

function calculate_gradient(f, params)

    g = []
    for par in params
        push!(g, similar(par))
        for i in eachindex(par)
            par[i] += 1e-8
            f_plus = f()
            par[i] -= 2e-8
            f_minus = f()
            g[end][i] = (f_plus - f_minus) / (2e-8)         #   finite difference formula  f (x + b) − f (x + a)
            par[i] += 1e-8
        end
    end

    return g
end

function train_model(epochs)
    nothing
end
