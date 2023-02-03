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
    images = shuffle(images)
    train_images = images[:, :, :, 1:train_size]
    test_images = images[:, :, :, train_size + 1:end]
    return train_images, test_images
end

function loadData(totalImages; ratio = 0.8)

    labels = shuffle(loadLabels())
    X_train, X_test = splitData(totalImages)

    y_train = labels[1: round(Int, size(totalImages, 4) * ratio)]
    y_test = labels[round(Int, size(totalImages, 4) * ratio) + 1: size(totalImages, 4)]

    return X_train, y_train, X_test, y_test

end
