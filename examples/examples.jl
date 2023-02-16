using Revise
import SeaTurtleID as ST

#We load the dataset from annotations.json

dataset = loadDataset()

#loading images with reshaped to 30x30
images = loadImages()

display(images[1])

#Converting  images to float and to Gray
processed_images = processImages(images)

display(processed_images[1])

#labels vectors of the turtles
labels = loadLabels()

#We get the training and test sets
X_train, y_train, X_test, y_test = loadSets()

Random.seed!(1234)
model = train(X_train, y_train, 10, 0.001, 1)
evaluate(model, X_train, y_train, X_test, y_test)
