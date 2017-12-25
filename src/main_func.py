from network_components.network import Network
import network_components.data_loader as loader

training_data, validation_data, test_data = loader.load_data()
net = Network([784, 30, 10])
net.train(training_data, 3, 30, 3, 3)
