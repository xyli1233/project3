#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sagemaker import get_execution_role

role = get_execution_role()
bucket = 'sagemaker-project3'


# In[8]:


role


# In[9]:


get_ipython().run_cell_magic('time', '', 'from sklearn import datasets  \nfrom sklearn.model_selection import train_test_split\ndigits = datasets.load_digits()  \n# First split: split the dataset into train_set and test_set\nX_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)\n\n# Second split: split the train_set into train_set and validation_set\nX_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)\n\nX_train = X_train.astype(\'float32\')\nX_val = X_val.astype(\'float32\')\nX_test = X_test.astype(\'float32\')\n\ny_train = y_train.astype(\'float32\')\ny_val = y_val.astype(\'float32\')\ny_test = y_test.astype(\'float32\')\n\n\ntrain_set = (X_train, y_train)\nvalidation_set = (X_val, y_val)\ntest_set = (X_test, y_test)\n# Print the shapes of the resulting sets\nprint(f\'Training set: {X_train.shape}\')\nprint(f\'Validation set: {X_val.shape}\')\nprint(f\'Test set: {X_test.shape}\')\n# import pickle, gzip, numpy, urllib.request, json\n\n# # urllib.request.urlretrieve("http://deeplearning.net/data/mnist.pkl.gz", "mnist.pkl.gz")\n# # with gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n# train_Set, valid_Set, test_set = pickle.load(digits, encoding=\'latinl\')\n')


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2, 10)

def show_digit(img, caption='', subplot = None):
    if subplot == None:
        _, (subplot) = plt.subplots(1, 1)
        imgr = img.reshape((8, 8))
        subplot.axis('off')
        subplot.imshow(imgr, cmap = 'gray')
        plt.title(caption)
# show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][300]))
index = 30  # The index of the digit image to show
digit_img = digits.images[index]
digit_label = digits.target[index]
show_digit(digit_img, f'This is a {digit_label}')


# In[30]:


from sagemaker import KMeans

data_location = 's3://{}/kmeans_highlevel_example/data'.format(bucket)
output_location = 's3://{}/kmeans_highlevel_example/output'.format(bucket)

print('training data will be updated to: {}'.format(data_location))
print('training artifacts will be uploaded to : {}'.format(output_location))

kmeans = KMeans(role = role,
               train_instance_count = 1,
                train_instance_type = 'ml.c4.xlarge',
                output_path = output_location,
                k=10,
                epoch = 150,
                data_location=data_location,
               )


# In[31]:


get_ipython().run_cell_magic('time', '', 'kmeans.fit(kmeans.record_set(train_set[0]))\n')


# In[32]:


get_ipython().run_cell_magic('time', '', "kmeans_predictor = kmeans.deploy(initial_instance_count = 1,\n                                instance_type='ml.m4.xlarge'\n                                )\n")


# In[33]:


get_ipython().run_cell_magic('time', '', "\nresult = kmeans_predictor.predict(validation_set[0][0:100])\nclusters = [r.label['closest_cluster'].float32_tensor.values[0] for r in result]\n")


# In[28]:


import numpy
for cluster in range(10):
    print('\n\n\nCluster {}:'.format(int(cluster)))
    digits = [img for l, img in zip(clusters, validation_set[0]) if int(l) == cluster]
    height = ((len(digits)-1)//5) + 1
    width = 5
    plt.rcParams["figure.figsize"] = (width,height)
    _, subplots = plt.subplots(height, width)
    subplots = numpy.ndarray.flatten(subplots)
    def show_digit2(img, title='', subplot=None):
        if subplot is None:
            plt.imshow(img.reshape(8,8), cmap=plt.cm.gray_r)
            plt.title(title)
        else:
            subplot.imshow(img.reshape(8,8), cmap=plt.cm.gray_r)
            subplot.axis('off')
            subplot.set_title(title)

    for subplot, image in zip(subplots, digits):
        show_digit2(image, subplot = subplot)
    for subplot in subplots[len(digits):]:
        subplot.axis('off')
    plt.tight_layout()
    plt.show()


# In[34]:


import numpy
for cluster in range(10):
    print('\n\n\nCluster {}:'.format(int(cluster)))
    digits = [img for l, img in zip(clusters, validation_set[0]) if int(l) == cluster]
    height = ((len(digits)-1)//5) + 1
    width = 5
    plt.rcParams["figure.figsize"] = (width,height)
    _, subplots = plt.subplots(height, width)
    subplots = numpy.ndarray.flatten(subplots)
    def show_digit2(img, title='', subplot=None):
        if subplot is None:
            plt.imshow(img.reshape(8,8), cmap=plt.cm.gray_r)
            plt.title(title)
        else:
            subplot.imshow(img.reshape(8,8), cmap=plt.cm.gray_r)
            subplot.axis('off')
            subplot.set_title(title)

    for subplot, image in zip(subplots, digits):
        show_digit2(image, subplot = subplot)
    for subplot in subplots[len(digits):]:
        subplot.axis('off')
    plt.tight_layout()
    plt.show()


# In[35]:


result = kmeans_predictor.predict(validation_set[0][230:231])
print(result)


# In[43]:


show_digit2(validation_set[0][20], 'This is a {}'.format(validation_set[1][20]))


# In[44]:


show_digit2(validation_set[0][230], 'This is a {}'.format(validation_set[1][230]))


# In[46]:


show_digit2(validation_set[0][110], 'This is a {}'.format(validation_set[1][110]))


# In[ ]:




