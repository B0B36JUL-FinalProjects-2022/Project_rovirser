using Flux
using Flux: onehotbatch

export toArray, splitData, loadData

function toArray(matrices)
    n = length(matrices)
    images = reshape(hcat(matrices...), (300, 300, 1, n))
    return images

end

function splitData(images, ratio=0.8)
    n = size(images, 1)
    train_size = round(Int, n * ratio)
    train_images = images[1:train_size, :, :, :]
    test_images = images[train_size + 1:end, :, :, :]
    return train_images, test_images
end

function loadData(imgs, turtles, num_turtles; onehot = true,  classes=1:400)

    X_train, X_test = splitData(imgs)
    y_train = turtles[1:round(Int, num_turtles*0.8)]
    y_test = turtles[round(Int, num_turtles*0.8)+1:num_turtles]

    if onehot
        y_train = onehotbatch(y_train, classes)
        y_test = onehotbatch(y_test, classes)
    end

    return X_train, y_train, X_test, y_test

end

