using Revise
import SeaTurtleID as turtle

loadDataset()
images = loadImages()
processImages(images)
detectFaces(max_pochs = 1000)

#First prediction with decisionTree

predTest = decisionTree()

turtle_counts = Dict{Int, Int}() # Initialize an empty dictionary

    # Iterate through the predictions
    for turtle in predTest
        if haskey(turtle_counts, turtle) # Check if turtle already in dictionary
            turtle_counts[turtle] += 1 # If so, increment count
        else
            turtle_counts[turtle] = 1 # If not, add turtle to dictionary with count 1
        end
    end

    turtle_counts = sort(turtle_counts)

    # Print out turtle counts
    for turtle in keys(turtle_counts)
        println("Turtle $(turtle) appears $(turtle_counts[turtle]) times")
    end

    total = 0
    for turtle in keys(turtle_counts)
        total += turtle_counts[turtle]
    end

    println(total)