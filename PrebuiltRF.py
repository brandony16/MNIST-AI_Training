from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from cacheMNIST import load_mnist_cached
import time

def main():
  # Load the MNIST dataset
  mnist = load_mnist_cached()
  X, y = mnist["data"], mnist["target"]

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # Normalize the data (pixel values to range [0, 1])
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  sample_size = 10000
  X_train_sample = X_train[:sample_size]
  y_train_sample = y_train[:sample_size]

  # Step 3: Implement the Random Forest Classifier
  # Initialize the RandomForest model
  clf = RandomForestClassifier(n_estimators=100, random_state=42)

  # Train the model
  clf.fit(X_train_sample, y_train_sample)

  # Step 4: Evaluate the Model
  # Predict the test set
  y_pred = clf.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy * 100:.2f}%")

  # Detailed classification report
  print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")