import argparse
from scipy import stats


def less_than_reject_null(t, p, alpha):
    if p / 2 < alpha and t < 0:
        return True
    return False


def greater_than_reject_null(t, p, alpha):
    if p / 2 < alpha and t > 0:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--means", nargs=2, type=float)
    parser.add_argument("--stds", nargs=2, type=float)
    parser.add_argument("--observations", nargs=2, type=int)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--test_type", "--test-type", default="greater-than", choices=("greater-than", "less-than"))
    args = parser.parse_args()

    t, p = stats.ttest_ind_from_stats(
        args.means[0], args.stds[0], args.observations[0],
        args.means[1], args.stds[1], args.observations[1],
        equal_var=False
    )

    print(t)
    print(p)
    if args.test_type == "greater-than":
        print(greater_than_reject_null(t, p, args.alpha))
    else:
        print(less_than_reject_null(t, p, args.alpha))




if __name__ == "__main__":
    main()
