module SeaTurtleID

using  JSON, DataFrames

# Write your package code here.
export loadDataset


function loadDataset()

    # Define the path to the JSON file
    file_path = "../annotations.json"

    # Read the JSON file
    data = JSON.parsefile(file_path)

    # Convert the JSON data to a dataset
    column_data = data["images"]

    dataset = DataFrame(column_data)

end

end
