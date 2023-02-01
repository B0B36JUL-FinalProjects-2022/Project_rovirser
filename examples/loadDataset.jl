using Revise
import SeaTurtleID as turtle

#We load the dataset from annotations.json

dataset = loadDataset()

#Show the columns
for i in 1:7
    dataset[:, i]
end