




Comparing ML Algorithms using Scikit-learn





Gyuho Tae
2020014348
Hanyang Univ. Artificial Intelligence CSE4007
2023.Nov.29

Abstract
-Problem setting
-Experiments
-Experimental results
-Discussion
-Conclusion

Problem setting
Statistics about the MNIST dataset shown in python:

This represents that max count label’s gap between min count label is less than 6%, so dissymmetry of the number of target’s labels aren’t that big enough.
I split them into training dataset and test dataset. Since there are only 1797 number of datasets, the portion of the test dataset will be about 20%. And this will be chosen in random order by using the code ‘from sklearn.model_selection import train_test_split’. And to compare among these models, I used random seed to shuffle orders. Since the digits’ data had repetitive pattern which may cause effect on the training session. We can see that data 0123456789 is repetitively appearing throughout the dataset. This shuffle was also chosen by random seed.

Experiments
In the experiments, models be, Linear Regression, Logistic Regression, SVM, Multilayer Perceptron, FDA. To test these models, I used the scikit-learn API. To briefly explain about these models’ algorithms, I’ll label them down below.
Linear Regression is model to assume relations be approximated as linear line. And to find best fitting linear equation. y=mx+c. To minimizes the difference between the predicted data and the actual answer value to find the optimal coefficients. Algorithm is for the adjustment of the value m and c in the way that best fits the given dataset.
Logistic Regression is model to classify classification problems. It uses the logistic function which is sigmoid function to model the probability of each class. So logistic regression equation is p=1/(1+exp(-mx-c). And because we’re classifying 10 digits, using softmax function is going to figure out which digit will be chosen. Softmax function computes probability of each class as sum of them to be 1. This makes multiclass classification. So, from the sklearn, made the experiments with 10000 iterations. 
Support Vector Machine (SVM) is model for classification. Support Vector Classifier (SVC) was used in the experiment and can perform multiclass classification. SVC uses various kernal functions to help create decision boundaries, like sigmoid, linear functions. Usually, SVC is effective in handling high dimension with smaller number of datasets.
Multilayer Perceptron is model made of neural network. It has neurons connected by weights and bias, which is learned by forward and backpropagation. In this model it can be designed by putting activation function, number of neurons and learning rate, etc. If no designated activation function and hyperparameters, adam is given as default, which is using ReLU as activation function and hidden layer of size of 100 and learning rate be 0.001. This is the state of when no setting is been made by. 

Experimental results
Every accuracy of the algorithms are follows. In the state of np.random.seed(14) and train_test_split(random_state=14). Numbers are rounded off to 4 decimal places.
Linear Regression: 0.6077, Logistic Regression: 0.9639, SVM: 0.9944, MLP: 0.9778, FDA: 0.9500, 
By looking at the results, accuracy ranking was given by SVM>MLP>Logistic Regression>FDA>Linear Regression. We can see that the SVM, MLP and Logistic regression and FDA are showing that those are well-defined models. However Linear Regression model is showing a definite limit in having good accuracy. Compare to other models. It is because linear regression is usually targeted for the continuous dataset and assumption of the linearity. So, digits data is given as classification problem and has non-linear data, so it couldn’t understand the complexity of the data. And linear regression accuracy is computed as R square value.
Logistic regression however, it is shown pretty good accuracy compared to the linear regression. It may be because it is made for the classification problems with high iterations. If it had just 10 iterations, accuracy goes down to 0.9444. Because of small data it still shows high accuracy but decreased compared to previous. Even if I increase the iteration, couldn’t get higher value than 0.9639. Then it shows that 0.0361 is the residual error.
For the discriminant model, we can see that numbers of data are mixed in the 2-dimension plot below. In LDA we can see that dividing lines are pretty much straight compared to the QDA lines. It shows that QDA fits more on the dataset, and since we know that the data is not linear, QDA may fits the data more accurately even if there is a risk of overfitting. But assuming that the data is much more complex, model should vary more. Also, the digits are pretty much similar with each other like 8 and 9 or 8 and 5. The reason why FDA had lower accuracy than the SVM is because it lowered the dimensions, so discriminant analysis works instead of using kernel. 
 
MLP is one of the models that can make accuracy as high as possible. If number of neurons are increased to (1000,1000,1000); previously (100,100,100), then the accuracy goes up to 0.9889. However, this takes too long time to learn (44sec), and this is not good especially when there is low number of datasets like this digit’s data. This is not a good characteristic because beside the SVM, it couldn’t show better accuracy nor the time duration (0sec). So, this dataset wasn’t good for MLP model.
And finally, SVM showed the best performance among these models. SVM is effective in non-linear, small data. And this is because it uses partial of the support vectors to find the boundary. Also, it has less risk of overfitting in order of having maximum margin between the datapoints near the boundary points. And using the kernel it made non-linear problem to linear problem, making higher dimensions data easily handled. 

Discussion
By looking at the experiment results, I could find that each of the model’s algorithms works well on the different conditions of datasets. For example, the fisher discriminant analysis is making 64-dimensional digits data to simple dimension which discard most of the data that were previously crucial in classifying in higher dimensions. And since Arabia numbers are look-alike each other, this may cause confusion when creating W.T matrix in fisher discriminant which makes maximum margin projection. Because of these problems, had to choose model that works on the high dimensions. MLP and SVM can be chosen. However, MLP’s network is more capable in computing more complex and bigger datasets. As I have shown previously, time duration shows significant difference between them.

Conclusion
It is shown that every model has its own pros and cons for different machine learning problems. And our goal is to decide which model may work and fit well on by seeing different type of data and classifying problems. In this MNIST problem we have seen that SVM was best among others, by having effectiveness in non-linear data, high-dimensional spaces, small size data, and handling multiclass classification. And we could have predicted it by looking at the digit data given.
