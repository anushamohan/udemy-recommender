# Personalized recommendation system for udemy users.

### Introduction 
Udemy is an online learning platform that gives the opportunity for the subject matter experts to create courses that are offered to the public. As of 2017, there are more than 55,000 courses on udemy. 

This project aims to explore different methods to build custom course recommendation system for udemy based on star rating and textual review combination based on similarity among different users and courses. 

### Motivation
I enroll in a course on udemy by checking out it's ratings and number of enrollments, and by watching some of it's preview and intro videos to get a feel for the course content. I basically spent lot of time before I decide to register in a course. Hence I decided to develop a personalized recommendation system and web app for udemy users.

### Work-flow

![alt text](https://user-images.githubusercontent.com/12536060/30229086-564c488e-9495-11e7-84d3-a6e495254834.png)

### Dataset
The dataset was obtained from Udemy API. Total of 330,000 ratings with text reviews were collected. The number of unique users in the dataset was 200,000 and number of unique courses was 976. 

![alt text](https://user-images.githubusercontent.com/12536060/30229388-90cf92b2-9496-11e7-889f-432134ae2c3f.png)

As seen from the above picture, the histogram is right skewed. More than 75% of users have taken or rated the course lesser than or equal to 2. The evaluation of ratings from these users was imposiible, hence these luke warm users were treated as cold start users and recommendation for these users was based on popularity model. 
The rating scale in the dateset is 0.5 - 5 as around 1000 users have given 0.5 rating to some courses. The utility matrix is very sparse with 975 courses and 12475 regular users and density of the matrix is around 0.4%

### Building Recommenders
##### 1) Popularity model
For the popularity model, the courses with higher review counts and ratings were selected. Using pearson correlation, the correlation between courses were calculated to display similar courses to the course that the user has picked. Courses with higher correlation to the target course were selected.

##### 2) Matrix Factorization model
Another important thing I did in this project was splitting the dataset manually into training, validation, and testing sets. To do this, the data was grouped by user id. The reason for doing this is that in each of the sets created, each user needs to be represented. Otherwise, it will be impossible to validate the predictions for that user

![alt text](https://user-images.githubusercontent.com/12536060/30230726-57443a74-949c-11e7-9b7a-717323d2f6fb.png)

For the regular user, the recommender system was built using matrix factorization model which gave better cross validated RMSE than the other models that were used in this project. The basemodel was predicting the test set using train set mean. The second basemodel was predicting the test set with train set mean after accounting for user bias and item bias. The item item collaborative filtering model was also tried, but did not give better RMSE than the base model as the utility matrix was very sparse.

For the matrix factorization model, two-fold cross validation was performed to make sure that model did not overfit and also to find the optimal parameters like number of latent factors, regularization constant to be used in the model.

##### 3) Matrix Factorization model with LDA
Latent Dirichlet Allocation (LDA) is a topic modelling algorithm that identifies the hidden sematic structues in a document.
According to LDA, documents are probability distributions over ltent topics and topics are probability distribution over words.  In the final model, I used LDA to extract latent features from text reviews and course topics. One challenge I had was choosing the number of latent features in text reviews and course topics. Whenever a user gives a review on a course, it was about video quality or intructor's knowlwdge on the course and course content. Hence I followed my intuition to go with 2 latent features. For the course topic, the number of latent features tried was from 4 to 20. But with 16, the latent features were more distinguishable and interpretable. After including these side data in the final matrix factorization model, the root mean squared error decreased by 3 percent. For the future work, I want to include user bias and item bias in the matrix factorization model and see whether the model performance improves. 




