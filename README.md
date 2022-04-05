# Room-Occupancy-Detection
Use Temperature, Relative Humidity, Light, CO2 and Humidity Ratio to detect room occupancy

Removed time as variable for the analysis

You can have a very first though that CO2 and Light would be the important factors for the prediction. Also, CO2 (human breathing) and lights could affect the temperature and therefore these three factors should be the most vital features

Below is the correlation among other variables

![Image of companies distribution](https://github.com/ccw0530/Room-Occupancy-Detection/blob/main/Correlation.png)

As 'Huminity' has relatively low correlation with 'Occupancy' and thus it is dropped in the dataset

Use below algorithms for learning the training dataset:
- Neural Network by Keras
- Logistic Regression
- Support Vector Classification
- K Nearest Neighbors

## Neural Network by Keras
Epoch 1/10
815/815 [==============================] - 4s 4ms/step - loss: 1.4282 - accuracy: 0.9321 - auc: 0.9473 - precision: 0.8158 - recall: 0.8785 - val_loss: 0.2132 - val_accuracy: 0.9775 - val_auc: 0.9837 - val_precision: 0.9453 - val_recall: 0.9959

Epoch 2/10
815/815 [==============================] - 3s 3ms/step - loss: 0.1446 - accuracy: 0.9656 - auc: 0.9890 - precision: 0.9109 - recall: 0.9289 - val_loss: 0.1863 - val_accuracy: 0.9516 - val_auc: 0.9861 - val_precision: 0.9305 - val_recall: 0.9372

Epoch 3/10
815/815 [==============================] - 3s 3ms/step - loss: 0.0899 - accuracy: 0.9774 - auc: 0.9934 - precision: 0.9352 - recall: 0.9601 - val_loss: 0.1067 - val_accuracy: 0.9771 - val_auc: 0.9912 - val_precision: 0.9453 - val_recall: 0.9949

Epoch 4/10
815/815 [==============================] - 3s 4ms/step - loss: 0.0818 - accuracy: 0.9808 - auc: 0.9927 - precision: 0.9372 - recall: 0.9751 - val_loss: 0.0997 - val_accuracy: 0.9775 - val_auc: 0.9925 - val_precision: 0.9453 - val_recall: 0.9959

Epoch 5/10
815/815 [==============================] - 3s 4ms/step - loss: 0.0767 - accuracy: 0.9792 - auc: 0.9935 - precision: 0.9314 - recall: 0.9740 - val_loss: 0.1096 - val_accuracy: 0.9779 - val_auc: 0.9916 - val_precision: 0.9454 - val_recall: 0.9969

Epoch 6/10
815/815 [==============================] - 3s 3ms/step - loss: 0.0694 - accuracy: 0.9827 - auc: 0.9944 - precision: 0.9387 - recall: 0.9826 - val_loss: 0.1174 - val_accuracy: 0.9764 - val_auc: 0.9904 - val_precision: 0.9426 - val_recall: 0.9959

Epoch 7/10
815/815 [==============================] - 3s 3ms/step - loss: 0.0652 - accuracy: 0.9804 - auc: 0.9940 - precision: 0.9238 - recall: 0.9890 - val_loss: 0.1025 - val_accuracy: 0.9779 - val_auc: 0.9916 - val_precision: 0.9454 - val_recall: 0.9969

Epoch 8/10
815/815 [==============================] - 3s 3ms/step - loss: 0.0653 - accuracy: 0.9827 - auc: 0.9935 - precision: 0.9325 - recall: 0.9902 - val_loss: 0.2017 - val_accuracy: 0.9700 - val_auc: 0.9808 - val_precision: 0.9264 - val_recall: 0.9969

Epoch 9/10
815/815 [==============================] - 3s 3ms/step - loss: 0.0637 - accuracy: 0.9804 - auc: 0.9941 - precision: 0.9225 - recall: 0.9907 - val_loss: 0.0867 - val_accuracy: 0.9722 - val_auc: 0.9921 - val_precision: 0.9481 - val_recall: 0.9774

Epoch 10/10
815/815 [==============================] - 3s 3ms/step - loss: 0.0547 - accuracy: 0.9838 - auc: 0.9945 - precision: 0.9333 - recall: 0.9948 - val_loss: 0.1268 - val_accuracy: 0.9775 - val_auc: 0.9909 - val_precision: 0.9488 - val_recall: 0.9918

## Logistic Regression
newton-cg Logistic Regression Training:  0.9860002456097261

newton-cg Logistic Regression Testing:  0.9786116322701689

newton-cg Logistic Regression AUC:  0.9917033099254008 

&nbsp;

lbfgs Logistic Regression Training:  0.9860002456097261

lbfgs Logistic Regression Testing:  0.9786116322701689

lbfgs Logistic Regression AUC:  0.991703917607967

&nbsp;

liblinear Logistic Regression Training:  0.9883335380081051

liblinear Logistic Regression Testing:  0.9782363977485928

liblinear Logistic Regression AUC:  0.9918394308202014

&nbsp;

sag Logistic Regression Training:  0.9884563428711777

sag Logistic Regression Testing:  0.9782363977485928

sag Logistic Regression AUC:  0.9917221480849492

&nbsp;

saga Logistic Regression Training:  0.986614269925089

saga Logistic Regression Testing:  0.9782363977485928

saga Logistic Regression AUC:  0.9915738735388273 

## Support Vector Classification
linear SVC Training:  0.9883335380081051

linear SVC Testing:  0.9782363977485928

linear SVC AUC:  0.9921499566114648

&nbsp;

rbf SVC Training:  0.9996315854107822

rbf SVC Testing:  0.9579737335834897

rbf SVC AUC:  0.980092926818004

&nbsp;

sigmoid SVC Training:  0.7876703917475132

sigmoid SVC Testing:  0.6352720450281426

sigmoid SVC AUC:  0.5 

## K Nearest Neighbors
K Nearest Neighbor Training:  0.9938597568463711

K Nearest Neighbor Testing:  0.9227016885553471

K Nearest Neighbor AUC:  0.9422902097477145 

&nbsp;

## Decision
Linear SVC gets the highest accuracy (97.82%) and AUC (99.21%) in testing data. As AUC is very close to 1 and means that this althorigm has quite high prediction ability
