using Revise
import SeaTurtleID as turtle

X_train, X_test, y_train, y_test = detectFaces()

m = Chain(
    Conv((2,2), 1=>16, sigmoid),
    MaxPool((2,2)),
    Conv((2,2), 16=>8, sigmoid),
    MaxPool((2,2)),
    flatten,
    Dense(288, size(y_train,1)),
    softmax,
)

file_name = joinpath("data", "model_prject.bson")
train_or_load!(file_name, m)

classes = 1:400
plts = []
for i in classes
    jj = 1:5
    ii = findall(onecold(y_train, classes) .== i)[jj]

    z1 = X_train[:,:,:,ii]
    z2 = m[1:2](X_train[:,:,:,ii])
    z3 = m[1:4](X_train[:,:,:,ii])

    kwargs = (nrows = 1, size = (600, 140))
    plot(
        imageplot(1 .- z1[:, :, 1, :], jj; kwargs...),
        imageplot(1 .- z2[:, :, 1, :], jj; kwargs...),
        imageplot(1 .- z2[:, :, end, :], jj; kwargs...),
        imageplot(1 .- z3[:, :, 1, :], jj; kwargs...),
        imageplot(1 .- z3[:, :, end, :], jj; kwargs...);
        layout = (5,1),
        size=(700,800)
    )
    savefig("Layers_$(i).svg")
end