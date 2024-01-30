# Chapter 2

Regression and classification

in this chapter going over single variable and multivariate regression formulas


define loss function and optimizer.

loss function defines quantity our model tries to minimize

optimizer decides minimization algorithm we're using.

```python 

# Normalize 

data = (data - data.min()) / (data.max() - data.min())

model = K.Sequentual([
                    Dense(1, input_shape = [1,], activation=None)
])
model.summary()

# Define loss

model.compile(loss='mean_squared_error', optimizer='sgd')


```

example 2 - one independent variable
example 3 - multivariate linear regression with multiple independent variables
example 4 - classification

classification tasks can be done by logistic regression 

