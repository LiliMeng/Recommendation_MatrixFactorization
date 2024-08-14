import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import argparse

def matrix_factorization_step(R, P, Q, K, alpha, beta):
    """
    Perform one step of matrix factorization using SGD.
    
    R: User-item rating matrix
    P: User-feature matrix
    Q: Item-feature matrix
    K: Number of latent features
    alpha: Learning rate
    beta: Regularization parameter
    """
    Q = Q.T
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                for k in range(K):
                    P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
    return P, Q.T

def calculate_rmse(matrix, P, Q):
    """
    Calculate RMSE for the matrix factorization.
    
    matrix: The original user-item rating matrix
    P: User-feature matrix
    Q: Item-feature matrix
    """
    Q = Q.T
    predicted = np.dot(P, Q)
    error = 0
    count = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0:
                error += (matrix[i][j] - predicted[i][j]) ** 2
                count += 1
    return np.sqrt(error / count)

def train_model(R, P, Q, K, steps, alpha, beta, log_file, test_matrix):
    """
    Train the matrix factorization model.
    
    R: Training user-item rating matrix
    P: User-feature matrix
    Q: Item-feature matrix
    K: Number of latent features
    steps: Number of iterations
    alpha: Learning rate
    beta: Regularization parameter
    log_file: Log file path
    test_matrix: Test user-item rating matrix for evaluation
    """
    with open(log_file, "w") as log:
        start_time = time.time()
        
        for step in range(steps):
            step_start_time = time.time()
            P, Q = matrix_factorization_step(R, P, Q, K, alpha, beta)

            if (step + 1) % 10 == 0:
                train_rmse = calculate_rmse(R, P, Q)
                test_rmse = calculate_rmse(test_matrix, P, Q)
                step_end_time = time.time()
                log.write(f"Step {step + 1} | Training RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Time: {step_end_time - step_start_time:.4f} seconds\n")
                log.flush()

        total_time = time.time() - start_time
        log.write(f"Total time taken for matrix factorization: {total_time:.4f} seconds\n")
    
    return P, Q

def load_movielens_data():
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(url, sep='\t', names=column_names)
    df = df.drop(columns=['timestamp'])
    return df

def prepare_data(df):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()

    train_matrix = np.zeros((num_users, num_items))
    for line in train_data.itertuples():
        train_matrix[line[1]-1, line[2]-1] = line[3]

    test_matrix = np.zeros((num_users, num_items))
    for line in test_data.itertuples():
        test_matrix[line[1]-1, line[2]-1] = line[3]
    
    return train_matrix, test_matrix, num_users, num_items

def evaluate_model(P, Q, train_matrix, test_matrix):
    """
    Evaluate the model on training and test sets.
    
    P: User-feature matrix
    Q: Item-feature matrix
    train_matrix: Training user-item rating matrix
    test_matrix: Test user-item rating matrix
    """
    train_rmse = calculate_rmse(train_matrix, P, Q)
    test_rmse = calculate_rmse(test_matrix, P, Q)
    
    print(f"Final Training RMSE: {train_rmse:.4f}")
    print(f"Final Test RMSE: {test_rmse:.4f}")

def main(K, steps, alpha, beta, log_filename):
    df = load_movielens_data()
    train_matrix, test_matrix, num_users, num_items = prepare_data(df)
    
    # Initialize matrices P and Q
    P = np.random.rand(num_users, K)
    Q = np.random.rand(num_items, K)
    
    # Train the model
    nP, nQ = train_model(train_matrix, P, Q, K, steps, alpha, beta, log_filename, test_matrix)
    
    # Evaluate the model
    evaluate_model(nP, nQ, train_matrix, test_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix Factorization on MovieLens 100K")
    parser.add_argument('--K', type=int, default=20, help='Number of latent features')
    parser.add_argument('--steps', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.02, help='Regularization parameter')
    parser.add_argument('--log_filename', type=str, default='training_log.txt', help='Log file name')
    
    args = parser.parse_args()
    
    main(K=args.K, steps=args.steps, alpha=args.alpha, beta=args.beta, log_filename=args.log_filename)
