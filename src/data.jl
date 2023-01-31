using Flux

export toArray, splitData, loadData

function toArray(matrices)
    n = length(matrices)
    h, w = size(matrices[1])
    images = Array{Float32, 4}(undef, n, h, w, 1)
    for i in 1:n
        images[i, :, :, 1] = matrices[i]
    end
    return images
end

function splitData(images, ratio=0.8)
    n = size(images, 1)
    train_size = round(Int, n * ratio)
    train_images = images[1:train_size, :, :, :]
    test_images = images[train_size + 1:end, :, :, :]
    return train_images, test_images
end

function loadData(imgs, turtles, num_turtles; onehot = true,  classes=0:9)

    X_train, X_test = split_data(imgs)
    y_train = turtles[1:round(Int, num_turtles*0.8)]
    y_test = turtles[round(Int, num_turtles*0.8)+1:num_turtles]

    return X_train, y_train, X_test, y_test

end

