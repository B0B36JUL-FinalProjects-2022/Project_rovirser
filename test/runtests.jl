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

    predictions = detectFaces(images)
    result = Vector{Int64}(undef, 1516)
    @test typeof(predictions) == typeof(result)
    @test length(predictions) == length(result)
    
end
