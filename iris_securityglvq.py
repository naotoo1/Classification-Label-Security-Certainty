import prototorch as pt
import pytorch_lightning as pl
import torch.utils.data
import prototorch.models
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from label_security1 import LabelSecurity
from protocert import ProtoCert
from sklearn.metrics import classification_report
from contour import Contourrn

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

# summary of glvq model certainty
print("glvq model certainty")
prototype_certainty_1 = protocert_1.my_proto_cert(y_test)
overall_1 = protocert_1.overall_model_cert(y_test)
print(prototype_certainty_1)
print(overall_1)
classes = ['class 0 (Iris Setosa)', 'class 1 (Iris Versicolour)', 'class 2 (Iris Virginica)']
print(classification_report(y_true=y_test, y_pred=y_pred_1, target_names=classes))

# Returns labels of data points whose computed classification label security is greater than or equal to
# a given thresh-hold(0.7)
index_listgl_0 = protocert_1.thresh_function(x=d1, y=0.7, y_='>=', y__='l', y___=None)

# Returns index of data points whose computed classification label security is greater than or equal to
# a given thresh-hold(0.7)
index_listgi_0 = protocert_1.thresh_function(x=d1, y=0.7, y_='>=', y__='i', y___=None)

# true label of data points whose classification label security is greater than or equal to the threshold security
true_labels_0 = protocert_1.thresh_y_test(x=index_listgi_0)
print(index_listgl_0)
print(true_labels_0)
print(classification_report(y_true=true_labels_0, y_pred=index_listgl_0, target_names=classes))


# summary parameter of glvq plot
# list containing  classification label securities greater than 0
thresh_list1 = protocert_1.thresh_function(x=d1, y=0, y_='>', y__='s', y___=None)

# list containing everything from(data index,predicted label, label security w.r.t the chosen thresh-hold)
thresh_list1_ = protocert_1.thresh_function(x=d1, y=0, y_='>', y__='all', y___=None)

print(thresh_list1, len(thresh_list1))
print(thresh_list1_)

# object for plot
contourn = Contourrn()

# summary of glvq plot
# plot learned  prototypes with train set

contourn.plot_dec_boundary(x=X_train, y=y_train, model=model_1, model_p=model_prototypes_1,
                           title='GLVQ Prototype Visualization', xlabel='Normalized sepal length',
                           ylabel='Normalized sepal width', model_type='GLVQ',
                           model_index=0)

# plot learned prototypes with the test set
contourn.plot_dec_boundary(x=X_test, y=y_test, model=model_1, model_p=model_prototypes_1,
                           title='GLVQ Prototype Visualization', xlabel='Normalized sepal length',
                           ylabel='Normalized sepal width', model_type='GLVQ',
                           model_index=0)

# contour plot with classification label securities of rejected and non rejected classification labels,prototypes and
# decision boundary
contourn.plot__newt(x=X_test, y=y_test, label_sec=thresh_list1, model_p=model_prototypes_1, index_list=thresh_list1_,
                    xlabel="Normalized sepal length",
                    ylabel="Normalized sepal width",
                    title="Iris test set classification label securities(GLVQ)", model_1=model_1,
                    model_type='GLVQ', model_index=0, h=0.7)


