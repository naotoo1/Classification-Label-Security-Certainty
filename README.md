# Classification-Label-Security-Certainty

This code contains the module (label_security1.py) for computing the classification label security with examples for GLVQ, GMLVQ and CELVQ models.

The models are first trained using a training data and tested on a test data.

So for every prediction from the models using a test data, the code returns the labels and their respective security.

The module is imported with the LabelSecurity Class, LabelSecurityM Class and LabelSecurityLM Class which is used to the compute the classification label security by calling on the methods in these Classes.

LabelSecurity Class for non matrix LVQ, LabelSecurityM for matrix LVQ and LabelSecurityLM for localized matrix LVQ

Examples are shown in the following python files (Iris_security_glvq.py, Iris_security_gmlvq.py and Iris_security_celvq.py)

The prerequisites needed for the code and outcome

LabelSecurity Class

Params = (test set, predicted labels of the test set, prototypes from the trained model using the train-set, fuzziness_parameter(default=2))

Methods

label_sec_f(x)

param x = predicted labels from the model using the test-set
outcome = classification labels and their respective securities

LabelSecurityM Class

Params = (test set, predicted labels of the test set, prototypes from the trained model using the train-set, omega_matrix from the trained model, fuzziness_parameter(default=2))

Methods

label_security_m_f(x)

param x = predicted labels from the model using the test-set
outcome = classification labels and their respective securities

LabelSecurityLM Class

Params = (test set, predicted labels of the test set, prototypes from the trained model using the train-set, List containing local omega_matrices from the trained model, fuzziness_parameter(default=2))

Methods

label_security_lm_f(x)

param x: predicted labels from the model using the test-set
outcome = classification labels and their respective securities



![gf](https://user-images.githubusercontent.com/82911284/165191983-dead7c3c-30b7-4f68-bc57-3e608df501bb.png)

![gmf](https://user-images.githubusercontent.com/82911284/165192166-f6cf594c-c50c-4ef8-9777-7636e954f94e.png)

![cf](https://user-images.githubusercontent.com/82911284/165192342-45d9fc5a-93d9-4d14-8be3-b2d281032af5.png)


