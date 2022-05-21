[![Python: 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch: 1.11](https://img.shields.io/badge/pytorch-1.11-orange.svg)](https://pytorch.org/blog/pytorch-1.11-released/)
[![Prototorch: 0.7.3](https://img.shields.io/badge/prototorch-0.7.3-blue.svg)](https://pypi.org/project/prototorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


# Classification-Label-Security-Certainty

## What is it?
Classification label security is prototype based recall proceedure that determines the extent to which predicted labels from LVQs classification results can be trusted.

## File structure

```
.
├── contour.py                               # Visualization of plots
├── iris_securitycelvq.py                    # Iris_test set example with celvq
├── iris_securityglvq.py                     # Iris_test set example with glvq
├── iris_securitygmlvq.py                    # Iris_test set example with gmlvq
├── iris_securitycelvq.py                    # Iris_test set example with celvq
├── optimised_m.py                           # script for optimal search of hyperparameter(m)
├── label_security1.py                       # classification label security/certainty for LVQs
├── protocert.py                             # Auxilliary code
└── README.md
```

## How to use?

```python
from label_security1 import LabelSecurity, LabelSecurityM, LabelSecurityLM 

# Non matrix LVQs
 label_security= LabelSecurity(x_test, class_labels, predict_results, model_prototypes, X)
 print(label_security.label_sec_f(y_pred))
```

```python
# Matrix and Local-Matrix LVQs
label_security= LabelSecurityM(x_test, class_labels, model_prototypes, model_omega, X)
print(label_security.label_security_m_f(y_pred))
```


The LVQ models are first trained using a training data. The learned prototypes are accessed and used to compute the classification label certainties of the test data.



## Visualization / Results

Classification results with reject and non-reject options based on the chow's approach (out of a simulated test results with a security thresh-hold of 0.7)  is shown below for the GLVQ, GMLVQ and CELVQ models respectively.

<p style='align:center'>
<img src='https://user-images.githubusercontent.com/82911284/165191983-dead7c3c-30b7-4f68-bc57-3e608df501bb.png'/>
</p>

<p style='align:center'>
<img src='https://user-images.githubusercontent.com/82911284/165192166-f6cf594c-c50c-4ef8-9777-7636e954f94e.png'/>
</p>

<p style='align:center'>
<img src='https://user-images.githubusercontent.com/82911284/165192342-45d9fc5a-93d9-4d14-8be3-b2d281032af5.png'/>
</p>

## Simulation

A simulated results from multiple reject thresholds for improving classification reliability using the CRT vs Chow is shown below for GLVQ

![Figure_4](https://user-images.githubusercontent.com/82911284/165868376-edfd75fd-dc67-41f1-a75a-305f2a72e06f.png)

 
 Below is a plot indicating a diminishing trend of the classification lable security of a sample data point with increasing m hyperparameter for ```label_security1.py```
 
 ![class_m](https://user-images.githubusercontent.com/82911284/167135470-8875729d-a0c6-4486-b623-2a31e7f23816.png)
 
 The optimal choice of hyperparameter m as against the default choice of m=2 is shown below for ```label_security1.py``` with the iris data set set using 
 **GLVQ**, **GMLVQ** and **CELVQ**.
 
 ![clsoptimedm](https://user-images.githubusercontent.com/82911284/167135492-888cca27-87a4-49d9-855e-6364910cd541.png)


## References

<a id="1">[1]</a> 
Fumera, G., Roli, F., & Giacinto, G. (2000, August). 
Multiple reject thresholds for improving classification reliability. 
In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 863-871). Springer, Berlin, Heidelberg.
