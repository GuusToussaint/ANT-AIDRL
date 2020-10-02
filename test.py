from ant import ANT
from ant.routers import FullyConnectedSigmoidRouter
from ant.transformers import FullyConnected1DTransformer
from ant.solvers import Linear1DSolver


t = ANT(in_shape=(10,), num_classes=3,
        new_router = FullyConnectedSigmoidRouter,
        new_transformer = FullyConnected1DTransformer,
        new_solver = Linear1DSolver)

while not t.root.fully_expanded():
    print(1)
    t.root = t.root.expand()
