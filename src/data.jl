using Random
using Statistics

export toArray, splitData, normalizeData, onehot, onecold, loadData

function toArray(matrices)
    n = length(matrices)
    images = reshape(hcat(matrices...), (100, 100, 1, n))
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

function normalizeData(X)
    mn = mean(X)
    st = std(X)

    return (X .- mn) ./ st
end

function onehot(y, classes)
    n = length(y)
    m = length(classes)
    result = zeros(n, m)
    for i in 1:n
        result[i, findfirst(classes .== y[i])] = 1
    end
    return result
end

onecold(y, classes) = [classes[argmax(col)] for col in eachcol(y)]

function loadData(totalImages; ratio = 0.8)

    labels = shuffle(loadLabels())
    X_train, X_test = splitData(totalImages)

    X_train = normalizeData(X_train)
    X_test = normalizeData(X_test)

    y_train = labels[1: round(Int, size(totalImages, 4) * ratio)]
    y_test = labels[round(Int, size(totalImages, 4) * ratio) + 1: size(totalImages, 4)]

    classes = unique(labels)

    y_train = onehot(y_train, classes)
    y_test = onehot(y_test, classes)

    return X_train, y_train, X_test, y_test

end
