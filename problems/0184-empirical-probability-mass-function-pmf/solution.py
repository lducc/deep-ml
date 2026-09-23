from collections import Counter
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    counter = Counter(samples)
    n = len(samples)
    answer = []
    for k, v in counter.items():   
        answer.append((k, v / n))

    return answer