
import prototorch as pt
import pytorch_lightning as pl
import torch.utils.data
import prototorch.models
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from label_security1 import LabelSecurity
from protocert import ProtoCert
from sklearn.metrics import accuracy_score
from crt import ThreshT
from procertnew import ProtoCertt
import numpy as np
import matplotlib.pyplot as plt

# Model 1
model_1 = pt.models.glvq.GLVQ(hparams=dict(distribution=[1, 1, 1]),
                              prototypes_initializer=pt.initializers.ZerosCompInitializer(2))

# Summary of model
print(model_1)

# Data_set and scaling
scaler = StandardScaler()
X, y = load_iris(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, y_train.shape)

# Dataset
train_ds = pt.datasets.NumpyDataset(X_train, y_train)
test_ds = pt.datasets.NumpyDataset(X_test, y_test)
print(X_train.shape)

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

type(train_loader)

x_batch, y_batch = next(iter(train_loader))
print(f"{x_batch=},{y_batch=}")

# Training
trainer_1 = pl.Trainer(max_epochs=100, weights_summary=None)

trainer_1.fit(model_1, train_loader)

# Get prototypes
print("glvq_prototypes", model_1.prototypes.numpy())

print("glvq_proto_labels", model_1.prototype_labels)

# Predict
y_pred_1 = model_1.predict(torch.Tensor(X_test))

# Summary of predict
print("glvq")
print(y_pred_1)

# Summary of test set labels
print(y_test)

# label security class params
m_1 = model_1.prototype_labels
predict_results_1 = y_pred_1
model_prototypes_1 = model_1.prototypes.numpy()

# Label_certainty
label_security_1 = LabelSecurity(x_test=X_test, class_labels=m_1, predict_results=y_pred_1,
                                 model_prototypes=model_prototypes_1, x_dat=X)

# Summary of classification label security for the test set
print("glvq")
d1 = label_security_1.label_sec_f(y_pred_1)
print(d1)

# summary of model certainty
protocert_1 = ProtoCert(y_test=y_test, class_labels=m_1, predict_results=y_pred_1)
protocert_1N = ProtoCertt(y_test=y_test, class_labels=m_1, predict_results=y_pred_1)

# simulation sets
simulation_list1 = np.arange(0, 0.2, 0.01)
simulation_list = np.linspace(0, 0.7, len(simulation_list1))


# A function to simulate the crt
def simulation(x):
    sim_list = []
    for i in x:
        th = ThreshT(y_test=y_test, class_labels=m_1, predict_results=y_pred_1, reject_rate1=i)
        h1 = th.thresh_new(d1, protocert_1, j=0)
        h2 = th.thresh_new(d1, protocert_1, j=1)
        h3 = th.thresh_new(d1, protocert_1, j=2)
        sim_list.append([h1, h2, h3])
    return sim_list


# simulate the crt securities
zz = simulation(simulation_list1)


# simulated accuracy rates which corresponds to the simulated rejection rates for crt.
def accuracy_crt():
    accuracy_list = []
    for i in range(len(zz)):
        non_rejected_labels = protocert_1N.thresh_function(x=d1, y=zz[i], y_='>=', y__='l', l3=[0, 1, 2])
        index_non_rejected_labels = protocert_1N.thresh_function(x=d1, y=zz[i], y_='>=', y__='i', l3=[0, 1, 2])
        true_labelsN = protocert_1N.thresh_y_test(x=index_non_rejected_labels)
        accuracy = accuracy_score(y_true=true_labelsN, y_pred=non_rejected_labels)
        accuracy_list.append(accuracy)
    return accuracy_list


# function to determine the rejection rate of chow
def rejection_rate(x, y1):
    z = len(x) / len(y1)
    return z


# function to determine the accuracy rates  and rejection rates of chow
def acc_rej_list_chow(x):
    accuracy_list = []
    rejection_rate_list = []
    for i in range(len(zz)):
        index_listgl = protocert_1.thresh_function(x=d1, y=simulation_list[i], y_='>=', y__='l', y___=None)
        index_listgi = protocert_1.thresh_function(x=d1, y=simulation_list[i], y_='>=', y__='i', y___=None)
        index_listglr = protocert_1.thresh_function(x=d1, y=0, y_='>', y__='l', y___=None)
        true_labels = protocert_1.thresh_y_test(x=index_listgi)
        accuracy = accuracy_score(y_true=true_labels, y_pred=index_listgl)
        rejection = rejection_rate(x=index_listgl, y1=index_listglr)
        rejection = 1 - rejection
        if rejection <= 0.2:
            rejection_rate_list.append(rejection)
            accuracy_list.append(accuracy)
    if x == 'accuracy_list':
        return accuracy_list
    if x == 'rejection_rate':
        return rejection_rate_list


# Summary results for CRT
print('CRT')
# summary accuracy for crt
print('simulated accuracy rate for crt')
print(accuracy_crt())
# summary rejection rate for crt
print('simulated rejection rates for crt')
print(simulation_list1)

# summary results for chow
print('chow method')
# summary accuracy rate for chow
print('simulated accuracy rates for chow')
print(acc_rej_list_chow(x='accuracy_list'))
print('simulated rejection rate for chow')
print(acc_rej_list_chow(x='rejection_rate'))

# visualize the performance of CRT vs Chow
plt.plot(simulation_list1, accuracy_crt(), label='CRT', marker='o')
plt.plot(acc_rej_list_chow(x='rejection_rate'), acc_rej_list_chow(x='accuracy_list'), label='Chow', marker='x')
plt.xlabel('Rejection %')
plt.ylabel('Accuracy %')
plt.legend()
plt.show()

