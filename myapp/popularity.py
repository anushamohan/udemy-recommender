import pandas as pd
import numpy as np

def popularity_courses(df):
    cold_users = df[df["user_review_count"] <=2]
    grouped_rating = pd.DataFrame(cold_users.groupby('published_title')['rating'].mean())
    grouped_rating['num of ratings'] = pd.DataFrame(cold_users.groupby('published_title')['rating'].count())
    grouped_rating.sort_values('num of ratings',ascending=False, inplace=True)
    grouped_rating.reset_index(inplace=True)
    top_courses = grouped_rating["published_title"].head(20)
    top_courses = pd.DataFrame(top_courses)
    top_courses = pd.merge(top_courses, df[["published_title", "price"]], on = "published_title")
    top_courses.drop_duplicates(inplace=True)
    return top_courses

def similar_popular(topic, df):
    if topic not in popularity_courses(df)["published_title"].unique():
        return "please enter a valid course name from the table above"
    cold_users = df[df["user_review_count"] <=2]
    grouped_rating = pd.DataFrame(cold_users.groupby('published_title')['rating'].mean())
    grouped_rating['num of ratings'] = pd.DataFrame(cold_users.groupby('published_title')['rating'].count())
    course_mat = cold_users.pivot_table(index='userid',columns='published_title',values='rating')
    topic_user_ratings = course_mat[topic]
    similar_to_topic = course_mat.corrwith(topic_user_ratings)
    corr_topic = pd.DataFrame(similar_to_topic,columns=['correlation'])
    corr_topic.dropna(inplace=True)
    corr_topic.sort_values('correlation',ascending=False, inplace=True)

    corr_topic= corr_topic.join(grouped_rating['num of ratings'])
    corr_topic.reset_index(inplace=True)
    similar_topic = corr_topic[corr_topic['num of ratings']>100].sort_values('correlation',ascending=False).head()
    similar_topic = pd.DataFrame(similar_topic)
    similar_topic = pd.merge(similar_topic, df[["published_title", "price"]], on = "published_title")
    similar_topic.drop_duplicates(inplace=True)
    return similar_topic[["published_title", "price"]]
