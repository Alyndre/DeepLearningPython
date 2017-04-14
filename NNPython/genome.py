import numpy

INPUT = 1
HIDDEN = 2
OUTPUT = 3


class NodeGene(object):

    def __init__(self, nid, bias, nodetype):
        self.nid = nid
        self.bias = bias
        self.nodetype = nodetype


class ConnectionGene(object):
    def __init__(self, inputid, outputid, weight, innovation, enabled):
        self.inputid = inputid
        self.outputid = outputid
        self.weight = weight
        self.innovation = innovation
        self.enabled = enabled


class Genome(object):

    def __init__(self):
        self.nodeGenes = []
        self.connectionGenes = []
        self.fitness = 0


def xorgenome():
    g = Genome()

    # INPUT NODES
    ng1 = NodeGene(1, numpy.random.rand(), INPUT)
    g.nodeGenes.append(ng1)
    ng2 = NodeGene(2, numpy.random.rand(), INPUT)
    g.nodeGenes.append(ng2)

    # HIDDEN NODES
    ng3 = NodeGene(3, numpy.random.rand(), HIDDEN)
    g.nodeGenes.append(ng3)
    ng4 = NodeGene(4, numpy.random.rand(), HIDDEN)
    g.nodeGenes.append(ng4)
    ng5 = NodeGene(5, numpy.random.rand(), HIDDEN)
    g.nodeGenes.append(ng5)
    ng6 = NodeGene(6, numpy.random.rand(), HIDDEN)
    g.nodeGenes.append(ng6)

    # OUTPUT NODES
    ng7 = NodeGene(7, numpy.random.rand(), OUTPUT)
    g.nodeGenes.append(ng7)
    ng8 = NodeGene(8, numpy.random.rand(), OUTPUT)
    g.nodeGenes.append(ng8)

    # CONNECTIONS
    cg1 = ConnectionGene(1, 3, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg1)
    cg2 = ConnectionGene(1, 4, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg2)
    cg3 = ConnectionGene(1, 5, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg3)
    cg4 = ConnectionGene(1, 6, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg4)

    cg5 = ConnectionGene(2, 3, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg5)
    cg6 = ConnectionGene(2, 4, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg6)
    cg7 = ConnectionGene(2, 5, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg7)
    cg8 = ConnectionGene(2, 6, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg8)

    cg9 = ConnectionGene(3, 7, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg9)
    cg10 = ConnectionGene(3, 8, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg10)

    cg11 = ConnectionGene(4, 7, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg11)
    cg12 = ConnectionGene(4, 8, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg12)

    cg13 = ConnectionGene(5, 7, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg13)
    cg14 = ConnectionGene(5, 8, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg14)

    cg15 = ConnectionGene(6, 7, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg15)
    cg16 = ConnectionGene(6, 8, numpy.random.uniform(low=-1, high=1), 1, True)
    g.connectionGenes.append(cg16)

    return g
