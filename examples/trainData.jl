using Revise
import SeaTurtleID as turtle

images = loadImages()
#We prepare all the data for the training sets
predictions = detectFaces(images)

turtle_counts = Dict{Int, Int}() # Initialize an empty dictionary

# Iterate through the predictions
for turtle in predictions
    if haskey(turtle_counts, turtle) # Check if turtle already in dictionary
        turtle_counts[turtle] += 1 # If so, increment count
    else
        turtle_counts[turtle] = 1 # If not, add turtle to dictionary with count 1
    end
end

# Print out turtle counts
for turtle in keys(turtle_counts)
    println("Turtle $(turtle) appears $(turtle_counts[turtle]) times")
end


plot(turtle_counts, Title="training Set", xlabel="TurtleID", ylabel="Counts", title="Test images",seriestype=:line)