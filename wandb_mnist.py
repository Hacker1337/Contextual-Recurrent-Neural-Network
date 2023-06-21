# Comet integration

import wandb

from ruamel.yaml import YAML

yaml=YAML(typ='safe')
with open('params.yaml', 'r') as file:
    config_file = yaml.load(file)


wandb.init(
    # set the wandb project where this run will be logged
    project="CRNN_mnist",
    
    # name="different-repeat-length",
    # notes="deterministic run",
    config=config_file,
)

config = wandb.config
# %%


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# %% [markdown]
# Load the MNIST dataset distributed with Keras. 

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

# %% [markdown]
# Filter the dataset to keep just the 3s and 6s,  remove the other classes. At the same time convert the label, `y`, to boolean: `True` for `3` and `False` for 6. 

# %%
def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

# %%
x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

# %%
x_train_small = tf.image.resize(x_train, (4,4)).numpy()
x_test_small = tf.image.resize(x_test, (4,4)).numpy()

# %%
import collections
def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass
    
    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of unique 3s: ", num_uniq_3)
    print("Number of unique 6s: ", num_uniq_6)
    print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))
    
    return np.array(new_x), np.array(new_y)

# %%
x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)

# %%
# batch_size = 10

# test_ds = tf.data.Dataset.from_tensor_slices((x_test_small, y_test)).batch(batch_size)
# train_ds = tf.data.Dataset.from_tensor_slices((x_train_nocon, y_train_nocon)).batch(batch_size)

# # %% [markdown]
# # ## Preparation.

# # %%
# # test run

# img1, labels = next(iter(train_ds))
# img2, labels = next(iter(test_ds))

# # %%
# for img in [img1, img2]:
#     plt.figure(figsize=(12, 6))
#     # assert(batch_size == 10)
#     for i in range(10):
#         plt.subplot(2, 5, i+1)
#         plt.title("3" if labels[i] else "6")
#         plot = plt.imshow(img[i], cmap='Greys', vmin=0, vmax=1)
#         # plt.axis("off")
#         plt.gca().get_xaxis().set_visible(False)
#         plt.gca().get_yaxis().set_visible(False)
#         # ax.get_yaxis().set_visible(False)
#     plt.show()


# %%
batch_size = 100

test_ds = tf.data.Dataset.from_tensor_slices((x_test_small, y_test)).batch(batch_size)
train_ds = tf.data.Dataset.from_tensor_slices((x_train_nocon, y_train_nocon)).batch(batch_size)

# %% [markdown]
# # Define model, loss, accuracy

# %%
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# %%
# accuracy_fn = tf.metrics.Accuracy()
def accuracy_fn(y_true, predicted_logits):
    matches = y_true == (predicted_logits > 0)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

# %%
# n = 10
# m = 1

# from model import CRNN
# model = CRNN(n, m)


# # %% [markdown]
# # ## Learning

# # %%
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# epochs = 50


# %% [markdown]
# Уменьшаю количество рекуррентных шагов сети

# %%
n = config["hidden_dims"]
m = config["input_size"]
from model import CRNN
model = CRNN(n, m, config["noise_factor"])

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
epochs = config["epochs"]


# %%
hist = {"train_loss":[], "test_loss":[], "train_accuracy":[], "test_accuracy":[], "train_batch_loss":[], "test_batch_loss":[]}
from tqdm import tqdm

pbar = tqdm(range(epochs), ncols=90)
for epoch in pbar:
    processed = 0
    hist["train_loss"].append(0)
    hist["train_accuracy"].append(0)
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model.process_img(x)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip([tf.clip_by_value(g, clip_value_min=-1, clip_value_max=1) for g in gradients], model.trainable_variables))
        acc = accuracy_fn(y, predictions)
        processed += y.shape[0]
        hist["train_loss"][-1] += loss
        hist["train_batch_loss"].append(loss)
        hist["train_accuracy"][-1] += acc*y.shape[0]
        pbar.set_description(f"Current loss: {loss/y.shape[0]:10.3f}  :::: "+
                            f"Current accuracy:\t {acc:.3f}")
        
    hist["train_loss"][-1] /= processed
    hist["train_accuracy"][-1] /= processed
    
    processed = 0
    hist["test_loss"].append(0)
    hist["test_accuracy"].append(0)
    for x, y in test_ds:
        predictions = model.process_img(x)
        loss = loss_fn(y, predictions)
        acc = accuracy_fn(y, predictions)
        processed += y.shape[0]
        hist["test_loss"][-1] += loss
        hist["test_batch_loss"].append(loss)
        hist["test_accuracy"][-1] += acc*y.shape[0]
        pbar.set_description(f"Current loss: {loss/y.shape[0]}  :::: "+
                            f"Current accuracy:\t {acc:.3f}")
        
    hist["test_loss"][-1] /= processed
    hist["test_accuracy"][-1] /= processed

    wandb.log({
        name: hist[name][-1] 
        for name in ["test_accuracy", "test_loss", "train_accuracy", "train_loss", ]
    })
    
wandb.run.log_code(include_fn=lambda path: path =="model.py" or path == "wandb_mnist.py")
# %%
print("Final test accuracy: \t", hist["test_accuracy"][-1].numpy())
print("Final train accuracy: \t", hist["train_accuracy"][-1].numpy())
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()   

