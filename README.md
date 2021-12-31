# ml_api_demo


This project was completed as part of the [Udacity Machine Learning DevOps Engineer NanoDegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821). And an initial project starter was forked from [here](https://github.com/udacity/nd0821-c3-starter-code-master). The project prompt is as follows:

```
Train a model that predicts if a person's salary will be above or below $50K based on a number of demographic factors. Then, use DVC, Github Actions, FastAPI and Heroku to build and deploy an API for serving predicts, with a full CI/CD process.
```


## ML Model

For information on the ML model, please see the [model card](https://github.com/kevinpmcgee14/ml_api_demo/blob/master/model/model_card.md)

## API

API can be accessed at https://km-salary-predictor.herokuapp.com/. API documentation can be found at  https://km-salary-predictor.herokuapp.com/docs

### Using the API

API is public, and can be accessed by anyone. The script [test_post.py](https://github.com/kevinpmcgee14/ml_api_demo/blob/master/test_post.py) can be run from the command line to make a request. Another option would be to use a product such as [Postman](https://www.postman.com/)