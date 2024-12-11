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
    import wandb
    file = 'data_utils/distance_captions.tsv'
    df = pd.read_csv(file, delimiter='\t')
    df.dropna(axis=0, how='any', inplace=True)
    labels = list(df['distance'])
    hist_data = [[] for _ in range (21)]

    # Put data in list of lists format for wandb histogram
    [hist_data[int(d)].extend([int(d)]) for d in labels if d < 21.0]
    # Filter empty lists
    col = [l for l in hist_data if len(l) > 0]

    table = wandb.Table(data=col, columns=["labeled_distances", "pred_distances"])
    wandb.log({'distance_labels': wandb.plot.histogram(table, "distances",
                                                    title="Distances in Labeled Data")})

    # Get list of values from labeled data
    # Make a plot using wandb
    # Make a function to retrieve distance preds from the testing function
    # Make a plot using wandb
    # Figure out how to compare the distributions.
    data =
    table = wandb.Table(data=data, columns=["Distance Labels"])
    wandb.log({'my_histogram': wandb.plot.histogram(table, "bird_scores",
                                                    title="Bird Confidence Scores")})

    hist_dataset = wandb.Histogram([all_examples])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_histogram()
    #process_captions()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
