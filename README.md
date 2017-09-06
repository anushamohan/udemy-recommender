Recommendation system for udemy users.

Motivation:
I choose courses on udemy by checking out ratings, number of enrollments for the courses, and by watching preview and intro videos. 
I basically spent lot of time to invest my money to enroll in a course on udemy. Hence I decided to develop a personalized 
recommendation system and web app for udemy users.


Data: The dataset was obtained from Udemy API. Total of 330,000 records (ratings) were collected with text reviews. The number of unique users in the dataset was 200,000 and number of unique courses was 976. Around >75% of users have taken or rated the course greater than or equal to 2. Hence the evaluation of ratings from these users was imposiible, these luke warm users were treated as cold start users and recommendation for these users will be based on popularity model. The rating scale in the dateset is 0 - 5 as around 1000 users have given 0.5 rating to some courses.


I split our dataset into training, validation, and testing sets. To do this I grouped the data by user id. The reason for doing this is that in each of these lists created, I wanted each user to be represented. Otherwise, it will be impossible to test the predictions for that user


For the regular users, I used matrix factorization model for recommender system. I performed k-fold cross validation to make sure the model does not overfit and also to find the optimal parameters line number of latent factors, regularization constant to be used in the model.

In the final model, I used Latent Dirichlet allocation to extract latent features from text revies and course topics. In order to choose the number of factors to be used by lda, for test review column, most of the reviews talked about course content, video quality/ instructor knowledge. Hence I decided to go with 2 topics for text reviews, whereas for course topics, with 16 latent features, the topics looked more distinguishable and interpretable. After including these side data in the final matrix factorization model, the root mean squared error decreased by 3 percent. 





