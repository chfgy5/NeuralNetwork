from neural_network import neural_network, np
import matplotlib.pyplot as plt

nn = neural_network()

# Run 1 epoch and print results:
# nn.train()
# print(np.around(nn.weights_hidden_layer, 4), end="\n\n")
# print(np.around(nn.weights_output_layer, 4), end="\n\n")
# print(np.around(nn.bias_hidden_layer, 4), end="\n\n")
# print(np.around(nn.bias_output_layer, 4), end="\n\n")
# print("%.4g" % round(nn.sum_of_squared_error(), 4))

# Run until change in Sum of Squared error is less than .001
SSE = nn.train(.001)

plt.figure(1)
plt.subplot(111)
plt.title('Sum of Squared Error per Epoch')
plt.ylabel('Sum of Squared Error')
plt.xlabel('Epoch')
plt.plot(SSE)

# Test against square [-2.1,  2.1]x[-2.1,  2.1] and plot results




# Run with different learning rates
SSE_07 = SSE

nn = neural_network()
nn.learning_rate = .01
SSE = nn.train(.001)
SSE_001 = SSE

nn = neural_network()
nn.learning_rate = .2
SSE = nn.train(.001)
SSE_02 = SSE

nn = neural_network()
nn.learning_rate = .9
SSE = nn.train(.001)
SSE_09 = SSE

plt.figure(2)
plt.subplot(411)
plt.title('Learning Rate Normal')
plt.ylabel('Sum of Squared Error')
plt.xlabel('Epoch')
plt.plot(SSE_07)

plt.subplot(412)
plt.title('Learning Rate .01')
plt.ylabel('Sum of Squared Error')
plt.xlabel('Epoch')
plt.plot(SSE_001)

plt.subplot(413)
plt.title('Learning Rate .2')
plt.ylabel('Sum of Squared Error')
plt.xlabel('Epoch')
plt.plot(SSE_02)

plt.subplot(414)
plt.title('Learning Rate .9')
plt.ylabel('Sum of Squared Error')
plt.xlabel('Epoch')
plt.plot(SSE_09)

plt.show()