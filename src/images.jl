using Images
using ColorTypes

export loadImages, processImages

function loadImages()

    dataset = loadDataset()
    images = dataset[:, "path"]
    num = length(images)

    # Add ./data/ to access the correct path
    for i in 1:num
        images[i] = "./data/"*images[i]
    end

    return images
end

function processImages(images :: Array{String})
    img_size = (300, 300)
    imgs = []
    for img in images
        resized_img = imresize(load(img), img_size)
        gray_img = Gray.(resized_img)
        push!(imgs, gray_img)
    end
    return imgs
end
