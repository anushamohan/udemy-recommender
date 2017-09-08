# Personalized recommendation system for udemy users.

### Introduction 
Udemy is an online learning platform that gives the opportunity for the subject matter experts to create courses that are offered to the public. As of 2017, there are more than 55,000 courses on udemy. 

This project aims to explore different methods to build custom course recommendation system for udemy based on star rating and textual review combination based on similarity among different users and courses. 

### Motivation
I enroll in a course on udemy by checking out it's ratings and number of enrollments, and by watching some of it's preview and intro videos to get a feel for the course content. I basically spent lot of time before I decide to register in a course. Hence I decided to develop a personalized recommendation system and web app for udemy users.

### Work-flow
![Alt text](/home/anusha/PicturesScreenshot from 2017-09-08 12-14-06.png?raw=true "Title")



### Data
The dataset was obtained from Udemy API. Total of 330,000 ratings with text reviews were collected. The number of unique users in the dataset was 200,000 and number of unique courses was 976. Around >75% of users have taken or rated the course lesser than or equal to 2. The evaluation of ratings from these users was imposiible, hence these luke warm users were treated as cold start users and recommendation for these users was based on popularity model. The rating scale in the dateset is 0.5 - 5 as around 1000 users have given 0.5 rating to some courses.




Another important thing I did in this project was splitting the dataset manually into training, validation, and testing sets. To do this I grouped the data by user id. The reason for doing this is that in each of the sets created, I wanted each user to be represented. Otherwise, it will be impossible to validate the predictions for that user

For the regular users, I used matrix factorization model for recommender system. I performed k-fold cross validation to make sure that model does not overfit and also to find the optimal parameters like number of latent factors, regularization constant to be used in the model.

In the final model, I used Latent Dirichlet allocation to extract latent features from text reviews and course topics. In order to choose the number of factors to be used by lda, in test review column, most of the reviews were about course content, video quality/ instructor knowledge. Hence I decided to go with 2 topics for text reviews, whereas for course topics, with 16 latent features, the topics looked more distinguishable and interpretable. After including these side data in the final matrix factorization model, the root mean squared error decreased by 3 percent. 





