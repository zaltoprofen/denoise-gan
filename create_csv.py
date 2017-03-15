import csv


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('list1', type=open)
    p.add_argument('list2', type=open)
    args = p.parse_args()

    fo = open('dataset.csv', 'w')
    w = csv.writer(fo)

    for x1, x2 in zip(args.list1, args.list2):
        x1, x2 = x1.strip(), x2.strip()
        w.writerow([x1, x2])

    fo.close()


if __name__ == '__main__':
    main()
