import numpy as np
import genome as g
import time

SIGMOID = 1
HYPERTAN = 2
NONE = 3


class Neuron(object):

    def __init__(self, nid, outputfunction, bias):
        self.nid = nid
        self.outputfunction = outputfunction
        self.bias = bias
        self.value = 0
        self.output = 0
        self.inputs = []
        self.outputs = []
        self.weights = {}
        self.fired = False
        self.derivative = 0

    def fire(self):
        if self.fired:
            return self.output
        else:
            if len(self.inputs) != 0:
                self.output = 0
                for neuron in self.inputs:
                    self.output += neuron.fire() * self.weights[neuron.nid]
                self.output += self.bias
                if self.outputfunction == SIGMOID:
                    self.output = sigmoid(self.output)
                elif self.outputfunction == HYPERTAN:
                    self.output = hypertan(self.output)
            else:
                self.output = self.value

            self.fired = True
            return self.output


class Network(object):

    def __init__(self, genome):
        self.genome = genome
        self.inputneurons = []
        self.outputneurons = []
        self.hiddenneurons = []
        self.neurons = {}

        for gene in genome.nodeGenes:
            if gene.nodetype == g.INPUT:
                n = Neuron(gene.nid, SIGMOID, 0)
                self.inputneurons.append(n)
                self.neurons[gene.nid] = n
            elif gene.nodetype == g.HIDDEN:
                n = Neuron(gene.nid, SIGMOID, 1)
                self.hiddenneurons.append(n)
                self.neurons[gene.nid] = n
            elif gene.nodetype == g.OUTPUT:
                n = Neuron(gene.nid, NONE, 0)
                self.outputneurons.append(n)
                self.neurons[gene.nid] = n

        for gene in genome.connectionGenes:
            n = self.neurons[gene.outputid]
            ni = self.neurons[gene.inputid]
            n.inputs.append(ni)
            n.weights[gene.inputid] = gene.weight
            ni.outputs.append(n)

    def evaluate(self, data):
        self.resetneurons()
        results = []
        for i, d in enumerate(data):
            self.inputneurons[i].value = d

        for n in self.outputneurons:
            results.append(n.fire())

        softmaxed = softmax(results)
        for i, d in enumerate(softmaxed):
            self.outputneurons[i].output = d

        return softmaxed

    def resetneurons(self):
        for key, value in self.neurons.iteritems():
            value.fired = False


def train(network, data, iterations, lr, momentum):

    for x in xrange(iterations):
        start = time.clock()

        for i, d in enumerate(data['d']):
            network.evaluate(d)
            expected = data['e'][i]
            for j, neuron in enumerate(network.outputneurons):
                neuron.derivative = expected[j] - neuron.output

            for neuron in network.outputneurons:
                calcNeuronDerivative(neuron.inputs)

            for neuron in network.outputneurons:
                for ni in neuron.inputs:
                    dw = 1*lr*neuron.derivative*ni.output
                    lw = neuron.weights[ni.nid]
                    neuron.weights[ni.nid] = lw + (dw * momentum)
                neuron.bias += 1*lr*neuron.derivative

            for neuron in network.hiddenneurons:
                for ni in neuron.inputs:
                    dw = 1*lr*neuron.derivative*ni.output
                    lw = neuron.weights[ni.nid]
                    neuron.weights[ni.nid] = lw + (dw * momentum)
                neuron.bias += 1*lr*neuron.derivative

        end = time.clock()
        print "Epoch: ", x, " in ", (end-start)
        pass


def calcNeuronDerivative(neurons):
    for neuron in neurons:
        sumk = 0
        for on in neuron.outputs:
            sumk += on.derivative * on.weights[neuron.nid]

        neuron.derivative = derivate(neuron.output) * sumk

    for neuron in neurons:
        if len(neuron.inputs) != 0:
            calcNeuronDerivative(neuron.inputs)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))  # axis=0


def sigmoid(x):
    w = 1. / (1 + np.exp(-x))
    return w


def hypertan(x):
    return np.tanh(x)


def derivate(x):
    return x*(1-x)
