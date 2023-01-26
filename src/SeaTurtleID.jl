module SeaTurtleID

using  JSON, DataFrames, Images, Flux

# Write your package code here.
export loadDataset, loadImages


function loadDataset()

    # Define the path to the JSON file
    file_path = "../data/annotations.json"

    # Read the JSON file
    data = JSON.parsefile(file_path)

    # Convert the JSON data to a dataset
    column_data = data["images"]

    dataset = DataFrame(column_data)

    return dataset

end

function loadImages()

    dataset = loadDataset()
    images = dataset[:, "path"]
    return images
end

function detectFaces()

    images = loadImages()
    turtlesID = collect(1:400)

    img_size = (224, 224)
    imgs = [Flux.Data.DataLoader(load(img), img_size) for img in images]


end

end
