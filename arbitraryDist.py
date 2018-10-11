import random
import argparse as ap
import functools
import sys
import matplotlib.pyplot as plt

def collectArgs():
    parser = ap.ArgumentParser(description='Returns a randomly sampled data set from an arbitrary probability distriution')
    parser.add_argument('numPoints', action='store', type=int, help='Number of datapoints to sample from the arbitrary distribution')
    parser.add_argument('probabilities', action='store', nargs='*', type=float, help='Specifies the probabilities of each bin in decimal form. Note: Must add up to 1')
    parser.add_argument('-r', '--resolution', type=float, default=0, action='store', help='Resolution of probabilities; If omitted data, will be normalized and rounding error may occur.')
    return parser.parse_args()



if __name__ == "__main__":
    args = collectArgs()
    numPoints = args.numPoints
    P = args.probabilities
    res = args.resolution

    total = functools.reduce(lambda x, y: x+y, P)
    print(total)
    
 ##   if total != 1:
 ##       print('ERROR: Probabilities do not sum to 1')
 ##       sys.exit(-1)

    data = []

    # Default resolution to the smallest probability in the list
    if res == 0:
        res = min(P)

    labels = range(0, len(P))
    P = [*map(lambda x, y = res: int(round(x/y, 0)), P)] #adding 0.5 fixes some truncation errors ehen casting to int
    for n in range(0, len(P)):
        for i in range(0, P[n]):
            data.append(labels[n])
    print(P)
    print(data)
        
    # Sample the data randomly
    sample = []

    for i in range(0, numPoints):
        sample.append(data[random.randint(0, len(data)-1)])

    print(sample)

    plt.figure()
    plt.hist(sample, len(labels))
    plt.show()
