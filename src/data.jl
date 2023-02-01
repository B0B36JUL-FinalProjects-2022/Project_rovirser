using Flux
using Flux: onehotbatch

export toArray, splitData, loadData

function toArray(matrices)
    n = length(matrices)
    images = reshape(hcat(matrices...), (30, 30, 1, n))
    return images

end

function splitData(images, ratio=0.8)
    n = size(images, 4)
    train_size = round(Int, n * ratio)
    train_images = images[:, :, :, 1:train_size]
    test_images = images[:, :, :, train_size + 1:end]
    return train_images, test_images
end

function loadData(totalImages, onehot; classes=1:400, ratio = 0.8)

    labels = loadLabels()
    X_train, X_test = splitData(totalImages)
    y_train = labels[1: round(Int, size(totalImages, 4) * ratio)]
    y_test = labels[round(Int, size(totalImages, 4) * ratio) + 1: size(totalImages, 4)]
    
    if onehot
        y_train = onehotbatch(y_train, classes)
        y_test = onehotbatch(y_test, classes)
    end

    return X_train, y_train, X_test, y_test

end
