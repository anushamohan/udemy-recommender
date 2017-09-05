import pandas as pd
import numpy as np


def category_courses(category, df):
    catIwant = df[df["course_category"] == category]
    grouped_cat = pd.DataFrame(catIwant.groupby('published_title')['rating'].mean())
    grouped_cat['num of ratings'] = pd.DataFrame(catIwant.groupby('published_title')['rating'].count())
    grouped_cat.sort_values('num of ratings',ascending=False, inplace=True)
    grouped_cat.reset_index(inplace=True)
    grouped_cat = grouped_cat["published_title"].head(10)
    cat_courses = pd.DataFrame(grouped_cat)
    cat_courses = pd.merge(cat_courses, df[["published_title", "price"]], on = "published_title")
    cat_courses.drop_duplicates(inplace=True)
    return cat_courses[["published_title", "price"]]
