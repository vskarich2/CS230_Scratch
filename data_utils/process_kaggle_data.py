import os

from data_utils.sample_coco2017_data import SampleCoco2017Data
from data_utils.sample_data import SampleData


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


if __name__ == '__main__':
    sd = SampleCoco2017Data(sample_size=5)
    sd.sample_the_data()


