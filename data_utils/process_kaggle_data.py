import os

def process_captions():
    TEST_IMG_COUNT = 10000
    count = 0
    read_filename = "/Users/vskarich/Downloads/Flickr8k_text/Flickr8k.token.txt"
    write_filename = "/Users/vskarich/Downloads/processed_captions_all.csv"

    try:
        os.remove(write_filename)
    except OSError:
        pass

    with open(read_filename) as file:
        lines = [line.rstrip() for line in file]

    lines = lines[:TEST_IMG_COUNT * 5]
    lines_dict = {}
    for line in lines:
        if count == TEST_IMG_COUNT:
            break
        split_line = line.split("#")
        img_id = split_line[0]
        caption = split_line[1][2:].replace(',', '')
        caption = caption.replace('?', '')
        caption = caption.replace('"', '')
        if img_id not in lines_dict:
            lines_dict[img_id] = caption
            count = count + 1
        else:
            if len(lines_dict[img_id]) > len(caption):
                lines_dict[img_id] = caption

    with open(write_filename, 'a') as the_file:
        the_file.write("img_id,caption" + '\n')
        for id, caption in lines_dict.items():
            the_file.write(id + "," + caption + '\n')

def create_histogram():
    import pandas as pd
    import matplotlib.pyplot as plt
    file = 'data_utils/distance_captions.tsv'
    df = pd.read_csv(file, delimiter='\t')
    df.dropna(axis=0, how='any', inplace=True)
    labels = list(df['distance'])






if __name__ == '__main__':
    create_histogram()
    #process_captions()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
