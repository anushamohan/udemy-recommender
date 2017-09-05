from flask import Flask
import pandas as pd
from flask import (request,
                   redirect,
                   url_for,
                   session,
                   render_template)

categories = ["sql", "programming", "web development", "art"]
from popularity import popularity_courses, similar_popular
from category import category_courses
import os

app = Flask(__name__)


@app.route('/', methods = ["GET", "POST"])
def welcome():
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        category = request.form['category']
        cat_course = pd.DataFrame(category_courses(category, df))
        cat_course.columns = ["Top courses", "Price"]
        return render_template('category.html', tables=cat_course.to_html(index=False))



@app.route('/popular', methods = ["GET", "POST"])
def popular():
    top_courses = popularity_courses(df)
    top_courses.columns = ["Top Courses", "Price"]
    if request.method == 'GET':
        return render_template('popular.html', tables = top_courses.to_html(index=False))
    elif request.method == 'POST':
        course = request.form['course']
        similar_course = similar_popular(course, df)
        similar_course.columns = ["Similar Courses", "Price"]
        return render_template('popular_similar.html', tables=similar_course.to_html(index=False))






if __name__ == "__main__":
    df = pd.read_csv("~/galvanize_notes/project/processed_data.csv")
    port = 5000
    app.run(host='0.0.0.0',port=port,debug=True)
