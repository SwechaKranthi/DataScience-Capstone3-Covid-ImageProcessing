# DataScience-Capstone3-Covid-ImageProcessing

*Given the Covid-19 Pandemic of the last 2 years, We can confidently say that there is still much more to learn of the disease.  Corona - COVID19 as a virus affects the respiratory system of healthy individual & Chest X-Ray is one of the important imaging methods to identify the corona virus. This project analyzes a Collection Chest X Ray of Healthy vs Pneumonia (Corona) affected patients infected patients along with few other categories such as SARS (Severe Acute Respiratory Syndrome ) ,Streptococcus & ARDS (Acute Respiratory Distress Syndrome). With this dataset, We develop a Neural Network Model to classify the X Rays of Healthy vs Pneumonia (Corona) affected patients. This resulting model powers the AI application to test the Corona Virus in Faster Phase.

## 1. Data

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. It is an environment that offers cool learning environments, data, and competitions for aspiring data scientists like me to grow. One such dataset is the coronahack-chest-xraydataset. With over 5800 images, this data set enables you to build a classification AI model for predicting covid patients from a series of X-rays. The data of which can be accessible through the following means below:


Joseph Paul Cohen. Postdoctoral Fellow, Mila, University of Montreal for the dataset below for corona dataset & 80% dataset collected from different sources.
> - [Kaggle Dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset)

Props to Joseph Paul Cohen. Postdoctoral Fellow, Mila, University of Montreal for the dataset below for corona dataset & 80% dataset collected from different sources: 
> - [Original Source](https://github.com/ieee8023/covid-chestxray-dataset)

## 2. Method

There are various ways to go about analyzing this data. Especially amongst the various tools and tricks available to data scientists ranging within AI, Neural Networks, Machine learning, It is important to classify the correct method for processing this data. So let’s break down the data first:

- What are we trying to predict?
  - We are trying to classify a series of images into its correct category of disease
- Supervised or Unsupervised?
  - First, Unsupervised learning is for grouping unlabeled data into distinct groups. Since we got labels for our data, We can use neural network to train and predict the correct labels. 
 - Thus our primary option ins supervised learning, but what kind?
- What kind of Supervised Learning?
  - There are 2 main types of Supervised Learning: Classification and Regression.
   - **Classification** is designed for classifying data (structured or unstructured) into distinct categories. Here, we are trying to assign a class, so this is our best tool. 
   - **Regression** is a type of supervised learning that helps us learn and understand the relationship between dependent and independent variables. Thus, we will not be using regression to predict the class of the images.

## 3. Data Cleaning 

[Pandas Profile Report](https://github.com/SwechaKranthi/DataScience-Capstone3-Covid-ImageProcessing/blob/main/Notebooks/Covid-MetaData-Profile-Report.html)


Data Cleaning is about 50% of the task of a Data Scientist. The accuracy of a predictive model depends highly on the quality of the data you feed it. In this step, we closely understand the data and verify what sort of cleaning procedures we need to perform to improve the quality of the data set. Now, there are many ways to check for the quality of data and how to clean it, but I can boil down my cleaning process in the following steps:

- Identify and Delete That Contain a Single Value
  - almost all columns had atleast 2 distinct values. 
- Consider Columns That Have Very Few Values
  - given that we are working with a meta data csv file, we arent missing any columns or have any tew few data
- Remove Columns That Have A Low Variance
  - This is an idea known as low variance filtering. As we know, Variance measures the dispersion of data realtive to its mean. Measuring the variance of different columns gives us an idea of the features that impact or not-impact our target variable - the sale price. To keep it simple, I decided to use 0 as the threshhold for low variance filter. As we know, None of the features had 0 variance or only 1 distinct value, so we didn't remove any columns. 
    - Normalizing the data is a pre-requesite to low variance filtering. It makes sure that no column is overvalued by reducing the range of all columns to a float between 0 and 1. This bring me to a potential mistake of my analysis. I standardized my data instead of normalizing due to the presence of different units of data. Standardization also works best under the assumption that the distribution of the data follows a normal bell-shaped curve. This is something we didn't check for and should be addressed in future modelling. 
- Identify and Delete that Contain Duplicate Data
  - There were no duplicates in the datasets provided by kaggle. This step can be avoided. 
- Address Missing Values 
  - I checked for counts of values that had a null value. In doing so I can determine what percentage of each feature was missing values. there were label columns that had missing values, we just filled in all missing values
- Dimensionality Reduction
  - this is a complication topic, so here is a reference to a good [article](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/) on it. In simple terms, If all features are created equal and we have 5000 feautures in a given data set, how do we determine which of those features are important in predicting our target value. There are many ways of doing this, but I used my two favorites: High Correlation Filtering, and Random Forest. 
    - we are gonna ignore this for image processing and just work on raw image data

## 4. EDA

 For EDA, we used pandas profiling to automate the process of exploring data. We generated a pandas profiling report that allows us to view the distributions of the data and see any missing values we needed to correct for. 
 
 [Pandas Profile Report for cleaned data](https://github.com/SwechaKranthi/DataScience-Capstone2-HousingPrices/blob/main/Reports/Housing_Data_Report_CleanedData.html)


## 5. Algorithms

For Modeling, I am using Tensorflow keras libraries. I am using sequential keras model. 

A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

We have a series of images and are determining labels, so this is perfect 1 to 1 tensor


## 6. Predictions

 [Pycaret Modeling Notebook](https://github.com/SwechaKranthi/DataScience-Capstone2-HousingPrices/blob/main/Notebooks/Housing_Modeling_pycaret.ipynb)
 
 [Predictions Data](https://github.com/SwechaKranthi/DataScience-Capstone3-Covid-ImageProcessing/blob/main/Notebooks/Covid-ImageProcessing.ipynb)
 
In the final predictions notebook, We have a working CNN model with a 77% accuracy on training set and a 68% accuracy on a validation set. 



## 7. Future Improvements

There are certainly many possibilities for improvements(This is one of my first NN, so lots of mistakes that can be corrected for):

- All hyperparameters were never touched or tuned. I could try more settings for the data augmentation and normalization. Also, adjusting the ratio between training and test data could also improve performance.

- Increasing the training set is the biggest one. Combining multiple data sources would make this project much better. In this dataset, the number of COVID-19 images is a bit low at 58 of ~5800 images. Adding more data should definitely improve the 77% identification rate.

- I could try a different achitectures and experiment with other Neural Networks more. 

- In the future, I would love to spend more time creating a cleaner data set. This is my first time working with images, I could improve my process by more closely understanding the data and optimizing images filter techniques.


## 8. Credits

Thank you to Raghunandhan Patthar for being an awesome Springboard Mentor. Thank you Joseph Paul Cohen. Postdoctoral Fellow, Mila, University of Montreal for the dataset for corona & 80% dataset collected from different sources. Thank you Praveen Govi for posting the data set to kaggle. 




Future trials:

try voting classifier systems

# Create empty lists that will storage the different weights

weights1 = []
weights2 = []
weights3 = []
scores = []

# Create a for loop to evaluate different combinations of weights

for i in np.arange(0.1,1, 0.1):
    for j in np.arange(0.1,1, 0.1):
        for k in np.arange(0.1,1, 0.1):
            clf_voting = VotingClassifier(estimators = [('est1', clf1), ('est2', clf2),
                                           ('est3', clf3)], voting = 'soft', weights = [i, j, k])
            clf_voting

