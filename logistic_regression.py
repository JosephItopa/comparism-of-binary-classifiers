# import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (np.random.rand(100) > 0.5).astype(int)

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Predictions and loss
y_pred_log_reg = log_reg.predict_proba(X)[:, 1]
loss_log_reg = log_loss(y, y_pred_log_reg)

print("Logistic Regression Loss:", loss_log_reg)