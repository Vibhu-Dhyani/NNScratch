import numpy as np

from utils import activationFunctionSigmoid, calculateLoss, derrSigmoid


class singleLayeredNetwork:
    # intialize
    def __init__(self):
        # weights
        self.w1 = self.w2 = self.w3 = self.w4 = self.w5 = self.w6 = np.random.normal()

        # bias
        self.b1 = self.b2 = self.b3 = np.random.normal()

    def feedForward(self, x):
        # define Neurons
        h1 = activationFunctionSigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = activationFunctionSigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = activationFunctionSigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def trainNN(self, data, all_y_trues):
        learningRate = 0.1
        epochs = 10000  # how many times we need to repeat on data

        for epoch in range(epochs):
            for x, yTrue in zip(data, all_y_trues):
                # output for h1
                h1_sum = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1Op = activationFunctionSigmoid(h1_sum)

                # output for h2
                h2_sum = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2Op = activationFunctionSigmoid(h2_sum)

                # output for o
                o1_sum = self.w5 * h1Op + self.w6 * h2Op + self.b3
                o1Op = activationFunctionSigmoid(o1_sum)
                yPred = o1Op

                # for w1 to w4 dl/dw = dl/dyPred * dyPred/dh * dh1/dw
                # for w5 and w6 dl/dw5 = dl/dyPred * dyPred/dw5

                dl_dyPred = -2 * (yTrue - yPred)

                # Neuron o1
                dyPred_dw5 = h1Op * derrSigmoid(o1_sum)
                dyPred_dw6 = h2Op * derrSigmoid(o1_sum)
                dyPred_db3 = derrSigmoid(o1_sum)

                dyPred_dh1 = self.w5 * derrSigmoid(o1_sum)
                dyPred_dh2 = self.w6 * derrSigmoid(o1_sum)

                # Neuron h1
                dh1_dw1 = x[0] * derrSigmoid(h1_sum)
                dh1_dw2 = x[1] * derrSigmoid(h1_sum)
                dh1_db1 = derrSigmoid(h1_sum)

                # Neuron h2
                dh2_dw3 = x[0] * derrSigmoid(h2_sum)
                dh2_dw4 = x[1] * derrSigmoid(h2_sum)
                dh2_db2 = derrSigmoid(h2_sum)

                # Calculate the dl/dw
                # Neuron h1
                dl_dw1 = dl_dyPred * dyPred_dh1 * dh1_dw1
                dl_dw2 = dl_dyPred * dyPred_dh1 * dh1_dw2
                dl_db1 = dl_dyPred * dyPred_dh1 * dh1_db1

                # Neuron h2
                dl_dw3 = dl_dyPred * dyPred_dh2 * dh2_dw3
                dl_dw4 = dl_dyPred * dyPred_dh2 * dh2_dw4
                dl_db2 = dl_dyPred * dyPred_dh2 * dh2_db2

                # Neuron o1
                dl_dw5 = dl_dyPred * dyPred_dw5
                dl_dw6 = dl_dyPred * dyPred_dw6
                dl_db3 = dl_dyPred * dyPred_db3

                # Tweaking variables
                # Neuron h1
                self.w1 -= learningRate * dl_dw1
                self.w2 -= learningRate * dl_dw2
                self.b1 -= learningRate * dl_db1

                # Neuron h2
                self.w3 -= learningRate * dl_dw3
                self.w4 -= learningRate * dl_dw4
                self.b2 -= learningRate * dl_db2

                # Neuron o1
                self.w5 -= learningRate * dl_dw5
                self.w6 -= learningRate * dl_dw6
                self.b3 -= learningRate * dl_db3

            if epoch % 10 == 0:
                ypreds = np.apply_along_axis(self.feedForward, 1, data)
                lossMSE = calculateLoss(all_y_trues, ypreds)
                print("Epoch %d loss: %.3f" % (epoch, lossMSE))


# Define dataset
data = np.array(
    [
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ]
)
all_y_trues = np.array(
    [
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ]
)

# Train our neural network!
network = singleLayeredNetwork()
network.trainNN(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3])  # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedForward(emily))  # 0.951 - F
print("Frank: %.3f" % network.feedForward(frank))  # 0.039 - M
