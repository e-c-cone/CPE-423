import argparse

def main():
    parser = argparse.ArgumentParser(description="BEEGUS")

    parser.add_argument("-t", "--transformer", default=False, action="store_true", help="Include transformer")
    parser.add_argument("-b", "--beegus", default=False, action="store_true", help="Testing")

    args = parser.parse_args()
    print(args.transformer, args.beegus)

if __name__ == "__main__":
    main()