import torch
from torch import nn 
import matplotlib.pyplot as plt

#Data preparing and loading 

# Create *known* parameters
weight = 0.7 
bias = 0.3 

#create data
start =0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim = 1)
Y = weight * X + bias
print(X[:10])
print(Y[:10])

# Create train/test split
train_split = int( 0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split :], Y[train_split :]
print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))

def plot_predictions(train_data=X_train, train_label=Y_train,
                     test_data=X_test, test_label=Y_test,
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10,7))
    # Plot training data in blue
  plt.scatter(train_data ,train_label , c ="b" , s= 4 , label = "Training Data")
  plt.scatter(test_data ,test_label , c ="g" , s= 4 , label = "Testing Data")
  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})
  plt.show()
  
plot_predictions()

# Create a Linear Regression model class

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1,dtype= torch.float) , requires_grad=True)
    self.bias = nn.Parameter(torch.randn(1,dtype= torch.float),  requires_grad=True)
    
      # Forward defines the computation in the model
  
  def forward(self , x: torch.tensor)->torch.tensor :
    return self.weights * x + self.bias
  

torch.manual_seed(44)
model_1 = LinearRegressionModel()
print(list(model_1.parameters()))

# Make predictions with model
with torch.inference_mode():
      y_preds = model_1(X_test)
      
# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")
plot_predictions(predictions=y_preds)

   

  
  




