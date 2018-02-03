from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# fn that returns trained linear function to make predictions
def train_and_test_linear_model(X_train, Y_train):
    LR = LinearRegression()
    LR.fit(X_train, Y_train)
    # return the prediction we've trained
    return LR.predict



# RIDGE REG
def train_and_test_ridge_model(alpha):
    def helper(X_train, Y_train):
        reg = linear_model.Ridge(alpha=alpha)
        reg.fit(X_train, Y_train)
        return reg.predict
    return helper


# LASSO REG
def train_and_test_lasso_model(alpha):
    def helper(X_train, Y_train):
        reg = linear_model.Lasso(alpha=alpha)
        reg.fit(X_train, Y_train)
        return reg.predict
    return helper


# ELASTICNET REG
def train_and_test_elastic_net_model(alpha):
    def helper(X_train, Y_train):
        reg = linear_model.ElasticNet(alpha=alpha)
        reg.fit(X_train, Y_train)
        return reg.predict
    return helper