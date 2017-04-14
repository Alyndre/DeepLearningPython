import network
import genome
import xordata

g = genome.xorgenome()

nn = network.Network(g)

network.train(nn, xordata.data, 5000, 1, 1.0)

for d in xordata.data['d']:
    print(nn.evaluate(d))
