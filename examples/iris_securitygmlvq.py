import prototorch as pt
import pytorch_lightning as pl
import torch.utils.data
import prototorch.models
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from label_security1 import LabelSecurityM
from protocert import ProtoCert
from sklearn.metrics import classification_report
from contour import Contourrn

# Model 2
model_2 = pt.models.glvq.GMLVQ(
    hparams=dict(input_dim=2, latent_dim=2, distribution=[1, 1, 1], proto_lr=0.01, bb_lr=0.01),
    prototypes_initializer=pt.initializers.ZerosCompInitializer(2))

# Summary of model
print(model_2)

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

trainer_2 = pl.Trainer(max_epochs=100, weights_summary=None)

trainer_2.fit(model_2, train_loader)

# Get prototypes

print("gmlvq_prototypes", model_2.prototypes.numpy())
print("gmlvq_proto_labels", model_2.prototype_labels)
print("gmlvq_omega", model_2.omega_matrix)

# Predict

y_pred_2 = model_2.predict(torch.Tensor(X_test))

# Summary of predict
print("gmlvq")
print(y_pred_2)

# Summary of test set labels
print(y_test)

# label security class params

m_2 = model_2.prototype_labels
predict_results_2 = y_pred_2
model_prototypes_2 = model_2.prototypes.numpy()
model_omega = model_2.omega_matrix

# Label_certainty
label_security_2 = LabelSecurityM(x_test=X_test, class_labels=m_2, model_prototypes=model_prototypes_2,
                                  model_omega=model_omega, x=X)

# Summary of classification label security for the test set
print("gmlvq")
d2 = label_security_2.label_security_m_f(y_pred_2)
print(d2)

# summary of model certainty
protocert_2 = ProtoCert(y_test=y_test, class_labels=m_2, predict_results=y_pred_2)

# summary of gmlvq model certainty
print("gmlvq model certainty")
prototype_certainty_2 = protocert_2.my_proto_cert(y_test)
overall_2 = protocert_2.overall_model_cert(y_test)
print(prototype_certainty_2)
print(overall_2)
classes = ['class 0 (Iris Setosa)', 'class 1 (Iris Versicolour)', 'class 2 (Iris Virginica)']
print(classification_report(y_true=y_test, y_pred=y_pred_2, target_names=classes))

# Returns labels of data points whose computed classification label security is greater than or equal to a
# given thresh-hold(0.7)
index_listgl_1 = protocert_2.thresh_function(x=d2, y=0.7, y_='>=', y__='l', y___=None)

# Returns index of data points whose computed classification label security is greater than or equal to a
# given thresh-hold(0.7)
index_listgi_1 = protocert_2.thresh_function(x=d2, y=0.7, y_='>=', y__='i', y___=None)

# true label of data points whose classification label security is greater than or equal to the threshold security
true_labels_1 = protocert_2.thresh_y_test(x=index_listgi_1)
print(index_listgl_1)
print(true_labels_1)
print(classification_report(y_true=true_labels_1, y_pred=index_listgl_1, target_names=classes))

# summary of gmlvq plot
# list containing  classification label securities greater than 0
thresh_list2 = protocert_2.thresh_function(x=d2, y=0, y_='>', y__='s', y___=None)

# list containing everything from(data index,predicted label, label security w.r.t the chosen thresh-hold)
thresh_list2_ = protocert_2.thresh_function(x=d2, y=0, y_='>', y__='all', y___=None)

print(thresh_list2, len(thresh_list2))
print(thresh_list2_)

# object for plot
contourn = Contourrn()

# summary of gmlvq plot
# plot learned  prototypes with train set
contourn.plot_dec_boundary(x=X_train, y=y_train, model=model_2, model_p=model_prototypes_2,
                           title='GMLVQ Prototype Visualization', xlabel='Normalized sepal length',
                           ylabel='Normalized sepal width', model_type='GMLVQ',
                           model_index=1)

# plot learned prototypes with the test set
contourn.plot_dec_boundary(x=X_test, y=y_test, model=model_2, model_p=model_prototypes_2,
                           title='GMLVQ Prototype Visualization', xlabel='Normalized sepal length',
                           ylabel='Normalized sepal width', model_type='GMLVQ',
                           model_index=1)

# contour plot with classification label securities, rejected and non rejected classification labels,prototypes
# and decision boundary
contourn.plot__newt(x=X_test, y=y_test, label_sec=thresh_list2, model_p=model_prototypes_2, index_list=thresh_list2_,
                    xlabel="Normalized sepal length",
                    ylabel="Normalized sepal width",
                    title="Iris test set classification label securities(GMLVQ)", model_1=model_2,
                    model_type='GMLVQ', model_index=1, h=0.7)
