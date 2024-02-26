Project Brief: 	Sentiment Analysis on E-commerce Dataset

Project Overview:
To determine the sentiments conveyed in customer reviews, we are performing a sentiment analysis project utilizing an e-commerce dataset. The goal is to create a model that can correctly categorize these evaluations' sentiments into positive, negative, or neutral. Data preprocessing, exploratory data analysis (EDA), model training, and the development of a web application for real-time sentiment analysis are all aspects of the project.

Dataset Details:
•	Source: The dataset is sourced from Kaggle and is available at the following link:  E-commerce Dataset.
•	Content: The dataset comprises various attributes such as name, brand, categories, primaryCategories, reviews.date, reviews.text, and reviews.title
•	Objective: Our primary goal is to analyze customer sentiments expressed in the reviews.text field of the dataset. This will help us understand the overall sentiment associated with different products and brands in the e-commerce domain

Proposed Actions:
•	Data Preprocessing: Handled missing values and duplicated data points to ensure data quality.
•	Exploratory Data Analysis (EDA): Explored the dataset to gain insights into the distribution of data, identify trends, and understand the composition of reviews across different attributes.

•	Class Imbalance Handling:  Addressed class imbalance in sentiment labels by applying appropriate techniques to balance the dataset, ensuring that the model is not biased towards the majority class.

•	Model Training: 
	Trained three machine learning models: Support Vector Classifier (SVC), Decision Tree, and Naive Bayes.
	Evaluated model performances using accuracy metrics.
	Chose the Support Vector Classifier (SVC) algorithm based on its higher accuracy compared to other models.

•	Text Vectorization:  Employed text vectorization techniques (e.g., TF-IDF) to convert the text-based reviews into numerical features that the machine learning models can understand.

•	Web Application Development: 
	Utilized Flask, a web framework, to create a user-friendly web application.
	Integrated the trained SVC model into the web application to enable real-time sentiment analysis of customer reviews.

Project Outcomes:
•	Model Performance: The Support Vector Classifier (SVC) demonstrated the highest accuracy of 64%, indicating its effectiveness in sentiment classification.



¬¬
























•	Web Application: The developed Flask-based web application allows users to input customer reviews and receive instant sentiment analysis results.



Supervisor Communication:
You can inform your supervisor that the project involves sentiment analysis on an e-commerce dataset sourced from Kaggle. The dataset contains attributes related to product information and customer reviews. The primary goal is to build a sentiment classification model that accurately categorizes customer sentiments as positive, negative, or neutral. The project includes data preprocessing, exploratory data analysis, class imbalance handling, model training, text vectorization, and the creation of a user-friendly web application for real-time sentiment analysis. The chosen model, SVC, achieved an accuracy of 64%, and the web application offers a practical interface for analyzing customer sentiments.
