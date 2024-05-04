
# find_all_matrics(): Calculate all the accuracy, precision, recall and f1-score
def find_all_metrics(true_positive, true_negative, false_negative, false_positive):

    # Calculate the precision
    precision = true_positive / (true_positive + false_positive)

    # Calcuate the recall
    recall = true_positive / (true_positive + false_negative)

    # Calculate the f1-score
    f1 = 2 * (precision * recall) / (precision + recall)

    # Calculate the accuracy
    accuracy = (true_positive + true_negative) / (true_positive + true_negative +
                                                  false_positive + false_negative)

    # Return all the rounded metrics
    return round(accuracy*100, 3), round(precision*100, 3), round(recall*100, 3), round(f1*100, 3)

# Print all the metrics
print(find_all_metrics(62, 106, 2, 1))
