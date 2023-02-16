import SeaTurtleID: turtle
using Test, DataFrames

@testset "SeaTurtleID.jl" begin
    # Write your tests here.

    df = DataFrame()
    data = loadDataset()
    @test typeof(data) == typeof(df)

    images = loadImages()
    @test typeof(images) == Vector{Matrix{RGB{N0f8}}}

    imagesProc = Vector{Matrix{Float32}}(undef, 7582)
    totalImages = processImages(images)
    @test typeof(totalImages) == typeof(imagesProc)  
    @test size(imagesProc) == size(totalImages)

    turtlesID = collect(1:7582)
    labels = loadLabels()
    @test typeof(loadLabels()) == typeof(turtlesID)
    @test length(turtlesID) == length(labels)

    array = toArray(totalImages)
    sol = Array{Float32, 4}
    @test ndims(imagesProc) == ndims(totalImages)

    X_train, y_train, X_test, y_test = loadSets()

    model2 = train(X_train, y_train, 10, 0.001, 10)
    @test typeof(model2) == Chain{Tuple{Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}, Conv{2, 4, typeof(relu), Array{Float32, 4}, Vector{Float32}}, MaxPool{2, 4}, var"#32#34", Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, typeof(softmax)}}
end
