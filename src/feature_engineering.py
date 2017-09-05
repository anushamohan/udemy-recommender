import numpy as np
import pandas as pd

"""
The function below extracts the name of the instructor from
instructor information column
"""
def instructor_title(x):
    return x.split(",")[3].split(":")[1]

"""
The function below extracts the job/title of the instructor from
instructor information column
"""

def instructor_job(x):
    d = {}
    y = x.split(",")
    for i in y:
        k = i.split(":")
        if len(k) < 2:
            continue
        d[k[0]] = k[1]
    if " u'job_title'" in d.keys():
        return d[" u'job_title'"]
    return None


def courseId(x):
    """
    Returns new course id generated and stored in the dictionary
    """
    return course_id_dic[x]

def userId(x):
    """
    Returns new user id generated and stored in the dictionary
    """
    return user_id_dic[x]


def categories(x):
    """
    Returns the categories that were manually assigned and stored in
    the dictionary
    """
    return assign_categories[x]

if __name__ == "___main__":

    df = pd.read_csv("my_pro.csv")
    """creates average rating per user column"""
    user_rating_avg = pd.DataFrame(df["rating"].groupby(df["userid"]).mean())
    user_rating_avg.columns = ["user_rating_avg"]
    user_rating_avg.reset_index(inplace=True)

    """creates number of ratings per user column"""
    user_rating_count = pd.DataFrame(df["rating"].groupby(df["userid"]).count())
    user_rating_count.columns = ["user_review_count"]
    user_rating_count.reset_index(inplace=True)

    """creates average rating per course column"""
    course_rating_avg = pd.DataFrame(df["rating"].groupby(df["course_id"]).mean())
    course_rating_avg.columns = ["course_rating_avg"]
    course_rating_avg.reset_index(inplace=True)

    """creates number of ratings per course column"""
    course_rating_count = pd.DataFrame(df["rating"].groupby(df["course_id"]).count())
    course_rating_count.columns = ["course_review_count"]
    course_rating_count.reset_index(inplace=True)

    df = pd.merge(df, user_rating_avg, on="userid")
    df = pd.merge(df, user_rating_count, on="userid")
    df = pd.merge(df, course_rating_count, on="course_id")
    df = pd.merge(df, course_rating_avg, on="course_id")

    df["instructor_name"] = df["visible_instructors"].apply(instructor_title)
    df["instructor_title"] = df["visible_instructors"].apply(instructor_job)

    courseids = sorted(df["course_id"].unique())
    userids = sorted(df["userid"].unique())

    new_courseids = range(1,len(courseids)+1)
    new_userids = range(1, len(userids)+1)

    df["new_course_id"] = df["course_id"].apply(courseId)
    df["new_user_id"] = df["userid"].apply(userId)

    df["created"] = pd.to_datetime(df["created"])

    assign_categories = {}
    for i in df.published_title.unique():
        print i
        """ manually assign a category based on the title of the course"""
        x = raw_input("enter the category")
        assign_categories[i] = x

    df["course_category"] = df["published_title"].apply(categories)

    df.to_csv("processed_data.csv")
