# Recommendation_MatrixFactorization

## Matrix Factorization
Matrix factorization is a popular technique used in recommendation systems to predict missing entries in a user-item matrix. Below is an example of how you can implement a matrix factorization method using stochastic gradient descent (SGD) on the MovieLens 100K dataset.

### 1. **Install Required Libraries**
First, make sure you have the necessary Python libraries installed:

```bash
pip install numpy pandas
```

### 2. **Load the MovieLens 100K Dataset**

```python
import pandas as pd
import numpy as np

# Load the MovieLens 100K dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=column_names)

# Drop the timestamp column as it's not needed for the matrix factorization
df = df.drop(columns=['timestamp'])
```

### 3. **Create the User-Item Matrix**

```python
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()

# Create the user-item matrix
user_item_matrix = np.zeros((num_users, num_items))

for line in df.itertuples():
    user_item_matrix[line[1]-1, line[2]-1] = line[3]
```

### 4. **Matrix Factorization Function using SGD**

```python
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    Perform matrix factorization using stochastic gradient descent.
    
    R: User-item rating matrix
    P: User-feature matrix
    Q: Item-feature matrix
    K: Number of latent features
    steps: Number of iterations
    alpha: Learning rate
    beta: Regularization parameter
    """
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:], Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T
```

### 5. **Train the Matrix Factorization Model**

```python
R = user_item_matrix
N = len(R)
M = len(R[0])
K = 20  # Number of latent features

# Initialize user and item latent feature matrices
P = np.random.rand(N, K)
Q = np.random.rand(M, K)

# Factorize the matrix
nP, nQ = matrix_factorization(R, P, Q, K)

# Reconstruct the full matrix
nR = np.dot(nP, nQ.T)
```

### 6. **Evaluate the Model**

After reconstructing the matrix, you can evaluate the model using metrics such as Root Mean Squared Error (RMSE).

```python
def rmse(predicted, actual):
    predicted = predicted[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(((predicted - actual) ** 2).mean())

print(f"RMSE: {rmse(nR, R)}")
```

### Explanation:

- **P and Q Matrices**: `P` is the user-feature matrix and `Q` is the item-feature matrix. Each row of `P` represents a user’s preferences, and each row of `Q` represents an item’s features.
- **Matrix Factorization**: The algorithm learns the matrices `P` and `Q` such that their dot product approximates the original user-item matrix.
- **Stochastic Gradient Descent (SGD)**: The matrix factorization is done using SGD, iteratively updating the matrices to minimize the error between the predicted and actual ratings.
- **Regularization**: `beta` is used to prevent overfitting by adding a penalty to large values in `P` and `Q`.

### Summary
This code demonstrates how to implement matrix factorization for the MovieLens 100K dataset using stochastic gradient descent. It’s a simple but powerful method to uncover latent features underlying the interactions between users and items, which can then be used to make recommendations.

## Code Explanation
The line of code you provided is part of a loop that iterates over the rows of a DataFrame using the `itertuples()` method. Here's an explanation of what each part of the code does:

### Code Breakdown

```python
for line in df.itertuples():
    user_item_matrix[line[1]-1, line[2]-1] = line[3]
```

1. **`df.itertuples()`**:
   - This method returns an iterator that yields named tuples containing the data from each row of the DataFrame `df`.
   - Each tuple returned by `itertuples()` has fields corresponding to the columns in the DataFrame. The first field is the index, followed by the values in each column.

2. **`for line in df.itertuples():`**:
   - This loop iterates over each row of the DataFrame `df`. 
   - `line` is a named tuple representing a row in the DataFrame.

3. **`line[1]`, `line[2]`, `line[3]`**:
   - `line[1]`: Represents the first column after the index, which in this case is the `user_id`.
   - `line[2]`: Represents the second column after the index, which is the `item_id`.
   - `line[3]`: Represents the third column after the index, which is the `rating`.
   - `line[0]` would be the index of the row (not used in this code).

4. **`user_item_matrix[line[1]-1, line[2]-1] = line[3]`**:
   - `user_item_matrix` is assumed to be a NumPy array or matrix where the rows correspond to users and the columns correspond to items.
   - The expression `line[1]-1` subtracts 1 from the `user_id` to adjust for 0-based indexing (since DataFrame indices are 1-based by default).
   - Similarly, `line[2]-1` subtracts 1 from the `item_id` for 0-based indexing.
   - The code then assigns the `rating` value (`line[3]`) to the corresponding position in `user_item_matrix` for the given `user_id` and `item_id`.

### Summary

This loop populates the `user_item_matrix` with the ratings from the DataFrame `df`. The matrix is indexed by user IDs (rows) and item IDs (columns), with the matrix entry at `[user_id-1, item_id-1]` set to the corresponding `rating`. The `-1` adjustment is necessary because Python uses 0-based indexing, while the IDs in the dataset are typically 1-based.


The `matrix_factorization` function performs matrix factorization using Stochastic Gradient Descent (SGD). This method is commonly used in recommendation systems to predict missing entries in a user-item interaction matrix. The function takes in several parameters and iteratively updates matrices `P` and `Q` to approximate the original matrix `R`. Here's a detailed breakdown of how the function works:

### Function Signature
```python
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
```

### Parameters

1. **`R` (Input Matrix)**:
   - This is the user-item interaction matrix. It could be a matrix of user ratings where `R[i][j]` represents the rating given by user `i` to item `j`.
   - The matrix `R` might have missing entries (e.g., places where a user has not rated an item), typically represented by zeros.

2. **`P` (User-Feature Matrix)**:
   - `P` is a matrix where each row corresponds to a user, and each column corresponds to a latent feature (i.e., a hidden characteristic inferred from the data).
   - The size of `P` is `num_users x K`, where `K` is the number of latent features.

3. **`Q` (Item-Feature Matrix)**:
   - `Q` is a matrix where each row corresponds to an item, and each column corresponds to a latent feature.
   - The size of `Q` is `num_items x K`, where `K` is the number of latent features.
   - The function transposes `Q` during computations, making it effectively `K x num_items`.

4. **`K` (Number of Latent Features)**:
   - This is the number of latent features used to represent users and items. It determines the dimensionality of the `P` and `Q` matrices.

5. **`steps` (Number of Iterations)**:
   - The number of iterations for which the algorithm runs. More iterations allow the model to converge more accurately but take more computational time.

6. **`alpha` (Learning Rate)**:
   - The learning rate for gradient descent. It controls the step size when updating the `P` and `Q` matrices.
   - A smaller learning rate may lead to slower convergence but can prevent overshooting, while a larger rate can speed up convergence but may cause the model to oscillate around the minimum.

7. **`beta` (Regularization Parameter)**:
   - This parameter controls the regularization, which helps prevent overfitting by penalizing large values in the `P` and `Q` matrices.
   - Regularization encourages the model to find simpler, more generalizable patterns.

### Function Logic

1. **Transpose `Q`**:
   - The function begins by transposing the `Q` matrix to simplify the dot product calculations that follow.

2. **Iteration Loop (`steps`)**:
   - The function iterates for the number of steps specified. During each iteration:
     - It loops over each element in the matrix `R`.
     - For each user-item pair `(i, j)` where `R[i][j]` is non-zero (i.e., the user has rated the item), the algorithm calculates the error between the actual rating `R[i][j]` and the predicted rating `P[i, :] * Q[:, j]`.
     - The error term `eij` is the difference between the actual rating and the predicted rating.

3. **Gradient Descent Updates**:
   <img width="714" alt="Screenshot 2024-08-14 at 11 37 15 AM" src="https://github.com/user-attachments/assets/07d823c1-5a95-40a5-8abe-bd1243e55c45">


4. **Compute the Error (`e`)**:
   - After updating the matrices, the function computes the total squared error of the matrix reconstruction. It also includes the regularization penalty.
   - If the error is sufficiently small (less than a threshold, here 0.001), the loop terminates early.

5. **Return the Factorized Matrices**:
   - After completing the specified number of iterations, or if the error criterion is met, the function returns the updated matrices `P` and `Q.T` (the transposed version of `Q`).

### Example Usage

Suppose you have a user-item rating matrix `R`, and you want to predict missing ratings using matrix factorization:

```python
R = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]])

N = len(R)
M = len(R[0])
K = 2  # Number of latent features

P = np.random.rand(N, K)
Q = np.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)

nR = np.dot(nP, nQ.T)
print(nR)
```

### Summary

- **Purpose**: The `matrix_factorization` function approximates a user-item rating matrix by decomposing it into two lower-dimensional matrices, `P` and `Q`, which represent the latent features of users and items.
- **Optimization**: The matrices are optimized using gradient descent, minimizing the error between the actual and predicted ratings.
- **Regularization**: Regularization is used to prevent overfitting by penalizing the magnitude of the latent features.
- **Output**: The function returns the factorized matrices, which can be used to predict missing ratings or analyze latent features in the data.
