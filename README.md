# project3

Introduction:
This project performs KMeans clustering on the MNIST dataset using Amazon SageMaker. The dataset is first split into a training set, validation set, and test set. The training set is used to train the KMeans model using SageMaker's high-level Python API. Once the model is trained, it is used to predict the clusters for the images in the validation set. The code then displays a sample of the images in each of the 10 clusters.


1. First, the code imports the required packages and defines the SageMaker execution role and S3 bucket that will be used to store the data and artifacts generated during the training.
<img width="606" alt="image" src="https://user-images.githubusercontent.com/123574025/227108450-705d7eee-c857-4a16-b1e0-1b42f3bfd3b9.png">

2. The code then loads the digits dataset from Scikit-learn and splits it into train, validation, and test sets using train_test_split.
<img width="993" alt="image" src="https://user-images.githubusercontent.com/123574025/227108617-48052c1d-69de-4161-a778-c7c8f62427b6.png">

3. next, the code defines a helper function show_digit that will be used to display individual digit images later on.
<img width="693" alt="image" src="https://user-images.githubusercontent.com/123574025/227108713-21c8c3cd-b0db-4536-8d04-fc313f3993c8.png">

4. The code then initializes a SageMaker KMeans estimator with the desired hyperparameters, including the number of clusters to be generated (k=10), the number of training epochs (epoch=150), and the instance type to use (train_instance_type='ml.c4.xlarge').
<img width="596" alt="image" src="https://user-images.githubusercontent.com/123574025/227108926-ee8246f4-3696-4216-9ed1-3526fc26b109.png">

5. The code then fits the KMeans model to the training data using the fit method.
<img width="670" alt="image" src="https://user-images.githubusercontent.com/123574025/227108974-826ae027-038f-465c-95cd-599479ef3ca3.png">


6. result:
<img width="575" alt="image" src="https://user-images.githubusercontent.com/123574025/227109155-453525d1-a419-4953-b5a4-2179902e9837.png">
<img width="553" alt="image" src="https://user-images.githubusercontent.com/123574025/227109186-f603e656-7e39-48ba-8c51-50c84cbc92b0.png">

<img width="626" alt="image" src="https://user-images.githubusercontent.com/123574025/227109208-da06818d-960b-4e99-b178-f6e37b548413.png">

<img width="584" alt="image" src="https://user-images.githubusercontent.com/123574025/227109241-3a7e2566-a105-43db-8ef5-502ef8e8d26a.png">

<img width="640" alt="image" src="https://user-images.githubusercontent.com/123574025/227109263-52bfc99c-538b-44ac-bb9c-6fcfc1320060.png">
<img width="569" alt="image" src="https://user-images.githubusercontent.com/123574025/227109396-d83935ac-4f96-4a1b-bdba-2d5038e8820c.png">

<img width="614" alt="image" src="https://user-images.githubusercontent.com/123574025/227109433-1019e73e-29a3-41ea-ac4c-4fb4a2db985c.png">
<img width="588" alt="image" src="https://user-images.githubusercontent.com/123574025/227109457-00813b13-9797-4220-8638-10c4a33e1769.png">
<img width="641" alt="image" src="https://user-images.githubusercontent.com/123574025/227109471-3bf2a14e-7066-438f-bb75-526ec907add2.png">
<img width="560" alt="image" src="https://user-images.githubusercontent.com/123574025/227109496-51ab43c7-a6ec-49b4-884a-9b8c194ca92f.png">


<img width="684" alt="image" src="https://user-images.githubusercontent.com/123574025/227109523-b5c6b7c3-496f-467b-99bb-ccdd460ba550.png">




