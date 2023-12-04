Link to mnist_train.csv & mnit_test.csv:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

************************************************************************************
To run the code:
    - Open a command prompt/terminal in the folder containing 'NeuralNetwork.py'
    - Run the command "python3 NeuralNetwork.py" to execute the file
    - Wait for execution, it might take few minutes.
    
************************************************************************************
Analysis:

When plotting train/test loss per epoch, we observed that deeper architectures
tend to converge to a lower loss than more shallow architectures at higher epochs.
However, deeper architectures may perform worse when trained for fewer epochs,
so it seems like deeper architectures need more epochs for training but fit better.
However, fitting better may result in over-fitting and lower validation accuracy,
as seen in the following results:

Validation Set Accuracy for different number of hidden layers:
    No hidden layers: 0.8921
    1 hidden layer: 0.9156
    2 hidden layers: 0.9088
    3 hidden layers: 0.9093

Ultimately, the best results come from 1 hidden layer,
even though the train loss may not be the best in this case.

************************************************************************************
