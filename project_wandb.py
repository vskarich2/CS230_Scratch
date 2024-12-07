import wandb

def log_test_predictions(
        images,
        labels,
        outputs,
        predicted,
        test_table,
        log_counter
):
  # obtain confidence scores for all classes
  scores = F.softmax(outputs.data, dim=1)
  log_scores = scores.cpu().numpy()
  log_images = images.cpu().numpy()
  log_labels = labels.cpu().numpy()
  log_preds = predicted.cpu().numpy()

  # adding ids based on the order of the images
  _id = 0
  for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
    # add required info to data table:
    # id, image pixels, model's guess, true label, scores for all classes
    img_id = str(_id) + "_" + str(log_counter)
    test_table.add_data(img_id, wandb.Image(i), p, l, *s)
    _id += 1
    if _id == NUM_IMAGES_PER_BATCH:
      break