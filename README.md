# Classification-Label-Security-Certainty

## Introduction
Classification label security determines the extent to which predicted labels from classification results can be trusted. The uncertainty surrounding classification labels is resolved by the security to which the classification is made. Therefore, classification label security is very significant for decision-making whenever we are encountered with a classification task.

## File structure

```
.
├── contour.py                               # Compiled files (alternatively `dist`)
├── iris_securitycelvq.py                    # Documentation files (alternatively `doc`)
├── iris_securityglvq.py                     # Source files (alternatively `lib` or `app`)
├── iris_securitygmlvq.py                    # Automated tests (alternatively `spec` or `tests`)
├── label_security1.py                       # Tools and utilities
├── protocert.py                             # 
└── README.md
```

## Implementation
This implementation investigates the determination of the classification label security by utilizing fuzzy probabilistic assignments of Fuzzy c-means
This code contains the module (label_security1.py) for computing the classification label security with examples for GLVQ, GMLVQ and CELVQ models.
The models are first trained using a training data and tested on a test data.

So for every prediction from the models using a test data, the code returns the labels and their respective security.

The module is imported with the **LabelSecurity Class** ,**LabelSecurityM Class** and **LabelSecurityLM Class** which is used to the compute the classification label security by calling on the methods in these Classes.

LabelSecurity Class for non matrix LVQ, LabelSecurityM for matrix LVQ and LabelSecurityLM for localized matrix LVQ

Examples are shown in the following python files (Iris_security_glvq.py, Iris_security_gmlvq.py and Iris_security_celvq.py)

The prerequisites needed for the code and outcome

Method_1

```Python
LabelSecurity Class

Params = (test set, predicted labels of the test set, prototypes from the trained model using the train-set, fuzziness_parameter(default=2))

label_sec_f(x)

param x = predicted labels from the model using the test-set
outcome = classification labels and their respective securities
```
Method_2
```Python
LabelSecurityM Class

Params = (test set, predicted labels of the test set, prototypes from the trained model using the train-set, omega_matrix from the trained model, fuzziness_parameter(default=2))

label_security_m_f(x)

param x = predicted labels from the model using the test-set
outcome = classification labels and their respective securities
```
Method_3
```Python
LabelSecurityLM Class

Params = (test set, predicted labels of the test set, prototypes from the trained model using the train-set, List containing local omega_matrices from the trained model, fuzziness_parameter(default=2))

label_security_lm_f(x)

param x: predicted labels from the model using the test-set
outcome = classification labels and their respective securities
```
## Visualization / Results
<p style='align:center'>
<img src='https://user-images.githubusercontent.com/82911284/165191983-dead7c3c-30b7-4f68-bc57-3e608df501bb.png'/>
</p>

<p style='align:center'>
<img src='https://user-images.githubusercontent.com/82911284/165192166-f6cf594c-c50c-4ef8-9777-7636e954f94e.png'/>
</p>

<p style='align:center'>
<img src='https://user-images.githubusercontent.com/82911284/165192342-45d9fc5a-93d9-4d14-8be3-b2d281032af5.png'/>
</p>


