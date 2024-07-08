from collections import Counter  # Import Counter from collections module

# Define a function to get the class distribution in the target variable Y
def get_class_dist(Y):
    # Use Counter to count the occurrences of each class in Y
    counter_resampled = Counter(Y)
    
    # Loop through the items in the counter
    for k, v in counter_resampled.items():
        # Calculate the distribution as a percentage
        dist = v / len(Y) * 100
        # Print the class, its count, and its distribution percentage
        print(f"Class={k}, n={v} ({dist:.2f}%)")