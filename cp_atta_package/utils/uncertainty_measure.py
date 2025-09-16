import numpy as np
from munch import Munch


"""
The value is range between 0 and 1
0 is perfect, 1 is worst.
"""
def brier_score(smxs, labels):
    
    if len(smxs.shape) == 1:
        smxs = np.expand_dims(smxs, axis=0)
    
    if len(labels.shape) == 0:
        labels = np.expand_dims(labels, axis=0)

    y_onehot = np.eye(smxs.shape[1])[labels]

    single_results = np.sum((smxs - y_onehot) ** 2, axis=1)
    
    return_value = {"single_results": single_results, 
                    "batch_result": np.mean(single_results).item()}
    
    return Munch(return_value)


"""
If the model is confident and correct (assigning a high probability to the true class), the NLL will be low. 
If the model is uncertain or incorrect (assigning a low probability to the true class), the NLL will be high.
The lowest value is 0 and the highest value is infinity.
"""
def nll(smxs, labels):
    if len(smxs.shape) == 1:
        smxs = np.expand_dims(smxs, axis=0)
    
    if len(labels.shape) == 0:
        labels = np.expand_dims(labels, axis=0)

    # avoid log(0)
    epsilon = 1e-15
    smxs = np.clip(smxs, epsilon, 1 - epsilon)  

    probs = smxs[np.arange(smxs.shape[0]) , labels]
    
    single_results = -1 * np.log(probs)

    return_value = {"single_results": single_results, 
                    "batch_result": np.mean(single_results).item() }
    
    return Munch(return_value)


def batch_expected_calibration_error(smxs, labels, M):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(smxs, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(smxs, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece[0].item()



def single_expected_calibration_error(smxs, labels):
    predicted_labels = np.argmax(smxs, axis=1)
    predicted_confidence = np.max(smxs, axis=1)
    is_correct = np.array([int(d1 == d2) for d1, d2 in zip(predicted_labels, labels)])
    calibration_error = np.abs(predicted_confidence - is_correct)
    return calibration_error

"""
Low ECE is good and high ECE is bad.
"""
def expected_calibration_error(smxs, labels, M):

    if len(smxs.shape) == 1:
        smxs = np.expand_dims(smxs, axis=0)
    
    if len(labels.shape) == 0:
        labels = np.expand_dims(labels, axis=0)

    return_value = {"single_results": single_expected_calibration_error(smxs, labels), 
                    "batch_result": batch_expected_calibration_error(smxs, labels, M)}
    
    return Munch(return_value)


"""
High Entropy: When the entropy is high, it means the softmax scores are more spread out 
(i.e., the model is less confident about its prediction). 
This typically indicates that the model is uncertain about which class the input belongs to.

Low Entropy: When the entropy is low, it means one of the softmax scores is much higher than the others 
(i.e., the model is more confident in its prediction). 
This indicates that the model is more certain about its decision.
"""
def entropy(smxs):

    if len(smxs.shape) == 1:
        smxs = np.expand_dims(smxs, axis=0)
    
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-12
    smxs = np.clip(smxs, epsilon, 1. - epsilon)
    entropy = -1 * np.sum(smxs * np.log(smxs), axis=1)
    
    return_value = {"single_results": entropy, 
                    "mean_entropy": np.mean(entropy).item(), 
                    "max_entropy": np.max(entropy).item(), 
                    "sum_entropy": np.sum(entropy).item() }
    
    return Munch(return_value)



def top_two_logit_diff(logits):

    # Find the top two logits using partition and indexing
    top_two = np.partition(logits, -2, axis=1)[:, -2:]
    
    # Calculate the difference between the highest and second-highest logits
    top_diff = top_two[:, 1] - top_two[:, 0]
    
    return_value = {"single_results": top_diff, 
                    "batch_result": np.mean(top_diff).item() }
    
    return Munch(return_value)