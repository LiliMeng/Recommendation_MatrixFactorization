import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import argparse
import matplotlib.pyplot as plt

def matrix_factorization_step(R, P, Q, K, alpha, beta):
    Q = Q.T
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                for k in range(K):
                    P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
    return P, Q.T

def calculate_rmse(matrix, P, Q):
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

def train(R, P, Q, K, steps, alpha, beta, log_file, eval_interval, test_matrix):
    with open(log_file, "w") as log:
        for step in range(steps):
            P, Q = matrix_factorization_step(R, P, Q, K, alpha, beta)

            if (step + 1) % eval_interval == 0:
                train_rmse = calculate_rmse(R, P, Q)
                test_rmse = calculate_rmse(test_matrix, P, Q)
                log.write(f"{step + 1},{train_rmse:.4f},{test_rmse:.4f}\n")
                log.flush()
    return P, Q

def plot_training_log(log_file):
    epochs = []
    train_rmse = []
    test_rmse = []

    with open(log_file, "r") as log:
        for line in log:
            epoch, train, test = line.strip().split(',')
            epochs.append(int(epoch))
            train_rmse.append(float(train))
            test_rmse.append(float(test))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_rmse, label="Training RMSE")
    plt.plot(epochs, test_rmse, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training and Evaluation RMSE over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_evaluation_curve.png")
    plt.show()

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

def main(K, steps, alpha, beta, log_filename, eval_interval):
    df = load_movielens_data()
    train_matrix, test_matrix, num_users, num_items = prepare_data(df)
    
    # Initialize matrices P and Q
    P = np.random.rand(num_users, K)
    Q = np.random.rand(num_items, K)
    
    # Train the model
    nP, nQ = train(train_matrix, P, Q, K, steps, alpha, beta, log_filename, eval_interval, test_matrix)
    
    # Plot the training log
    plot_training_log(log_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix Factorization on MovieLens 100K")
    parser.add_argument('--K', type=int, default=20, help='Number of latent features')
    parser.add_argument('--steps', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.02, help='Regularization parameter')
    parser.add_argument('--log_filename', type=str, default='training_log.txt', help='Log file name')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval in epochs')
    
    args = parser.parse_args()
    
    main(K=args.K, steps=args.steps, alpha=args.alpha, beta=args.beta, log_filename=args.log_filename, eval_interval=args.eval_interval)
