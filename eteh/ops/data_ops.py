
def my_cat(file_path):
    # read text lines from a text file, like the cat in unix
    with open(file_path) as file:
        lines = [line.rstrip('\n') for line in file]
    return lines