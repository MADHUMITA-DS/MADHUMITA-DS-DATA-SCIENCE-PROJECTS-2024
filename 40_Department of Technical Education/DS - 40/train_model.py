from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv(r'C:\Users\user\Downloads\IT.csv')  # Adjust this path as needed

# Print the column names to check for 'Rank_Category'
print("Columns in DataFrame:", df.columns)

# Create a new target column based on 'Rank' (example logic)
# Here we are creating a simple categorization, you can modify this logic as needed
def categorize_rank(rank):
    if rank <= 100:  # Change this threshold based on your criteria
        return 'High'
    elif rank <= 200:
        return 'Medium'
    else:
        return 'Low'

# Apply the categorization function
df['Rank_Category'] = df['Rank'].apply(categorize_rank)

# Function to split the dataset
def split_data(df):
    X = df[['Rank']]  # Features (make sure 'Rank' exists in your dataset)
    y = df['Rank_Category']  # Your target column
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = split_data(df)

# Train the model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Save the model to a file
joblib.dump(clf, 'model.pkl')  # This line saves your trained model to a file named 'model.pkl'
