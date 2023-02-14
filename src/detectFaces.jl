using Flux
using Flux: params
using Flux: flatten
using Statistics
using Plots
using Flux.Data: DataLoader


# Write your package code here.
export detectFaces, loss, accuracy, Adam, calculate_gradient, evaluate

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
    n = size(y, 2)
    return -sum(X .* log.(y)) / n
end

function accuracy(X, y, classes)
    return mean(onecold(model(X), classes) .== onecold(y, classes))
end

"""
    Adam(model;  learning_rate = 0.001, epochs=10, decay_rate1=0.9, decay_rate2=0.999, ε=1e-8)

update rule computes an update for the j-th parameter based on the gradient, the moving average, and the moving variance of the gradient, 
and adjusts the parameter by subtracting this update from the original parameter.

Return: updated model

"""

function train_sgd(model, X_train, y_train, X_test, y_test; num_epochs=10, learning_rate=0.1)

    parameters = params(model)
    
    for epoch in 1:num_epochs        
       
        # Forward pass
        prediction = model(X_train)
        
        # Compute the loss
        loss = Flux.crossentropy(prediction, y)
        
        # Backward pass
        grads = gradient(() -> loss, parameters)
        
        # Update parameters
        for j in 1:length(params)
            parameters[j] -= learning_rate * grads[j]
        end
    
        # Compute accuracy on the test set
        test_predictions = model(X_test)
        accuracy = mean(argmax(test_predictions, 1) .== argmax(y_test, 1))
        
        println("Epoch: $(epoch), Accuracy: $(accuracy)")
    end

end

function train_adam(model, X_train, y_train, X_test, y_test; epochs = 10, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)

    m = size(y_train, 2)     #number of training examples
    parameters = Flux.params(model)
    num_params = length(parameters)
    t = 0       # time step that keeps track of the number of iterations in the optimization process

    # Initialize the parameters for the first and second moments of the gradients with zero tensors
    m_params = [zeros(size(p)) for p in parameters]
    v_params = [zeros(size(p)) for p in parameters]

    # Loop over the specified number of epochs
    for epoch in 1:1
        for i in 1:1
            x = X_train[:, :, :, 1]
            y = y_train[:, 1] 
            grads = gradient(() -> loss(model(X_train), y_train), parameters)
            t += 1

            # Update the first and second moment
            m_params = [beta1 * m_param + (1 - beta1) * grad for (m_param, grad) in zip(m_params, grads)]
            v_params = [beta2 * v_param + (1 - beta2) * grad.^2 for (v_param, grad) in zip(v_params, grads)]

            # Caluclate the bias first and second moment
            m_hat = [m_param / (1 - beta1^t) for m_param in m_params]
            v_hat = [v_param / (1 - beta2^t) for v_param in v_params]

            # Update the parameters
            for j in 1:1
                parameters[j] -= learning_rate * m_hat[j] / (sqrt.(v_hat[j]) .+ epsilon)
            end
        end

        accuracy = accuracy(model, X_test, y_test)
        println("Epoch $epoch, Accuracy: $accuracy")
    end

    return model
end


function evaluate(X_test, y_test, classes)

    println("Train accuracy = ", accuracy(X_train, y_train, classes))
    println("Test accuracy = ", accuracy(X_test, y_test, classes))

end
