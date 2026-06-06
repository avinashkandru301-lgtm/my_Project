# Image Restoration Transformer
# Converted from Jupyter Notebook. Outputs removed.


# %% Cell 1
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% Cell 2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %% Cell 3
data_dir = "/kaggle/input/reside-6k/RESIDE-6K/train"
noise_dir = "/kaggle/input/multi-noises-for-image-denoising/dataset"
low_light_dir = "/kaggle/input/lol-low-light-dataset/lol_dataset/our485"

# %% Cell 4

hazy_dir = os.path.join(data_dir, "hazy")
clear_dir = os.path.join(data_dir, "GT")

noisy_dir = os.path.join(noise_dir, "noises" , "multiplicative")
original_dir = os.path.join(noise_dir, "original")

low_light_high = os.path.join(low_light_dir, "high")
low_light_low = os.path.join(low_light_dir, "low")

IMG_SIZE = 128

# %% Cell 5
hazy_dir = os.path.join(data_dir, "hazy")
clear_dir = os.path.join(data_dir, "GT")

noisy_dir = os.path.join(noise_dir, "noises", "multiplicative")
original_dir = os.path.join(noise_dir, "original")

low_light_high = os.path.join(low_light_dir, "high")
low_light_low = os.path.join(low_light_dir, "low")

# Function to collect paired (input, target) image paths
def collect_pairs(input_dir, target_dir):
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    target_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    
    # Ensure both have same length
    n = min(len(input_files), len(target_files))
    pairs = [(os.path.join(input_dir, input_files[i]), os.path.join(target_dir, target_files[i])) for i in range(n)]
    return pairs

# Collect all datasets
hazy_pairs = collect_pairs(hazy_dir, clear_dir)
noisy_pairs = collect_pairs(noisy_dir, original_dir)
lowlight_pairs = collect_pairs(low_light_low, low_light_high)

# Merge all into one dataset
all_pairs = hazy_pairs + noisy_pairs + lowlight_pairs

print(f"Total dataset size: {len(all_pairs)}")
print("Sample pair:", all_pairs[0])

# Example: load and visualize one sample
num_samples = 1  # number of samples per category to visualize

# ---------- 1. Hazy → Clear ----------
print("Hazy → Clear samples")
for i in range(num_samples):
    inp_path, tar_path = hazy_pairs[i]
    
    inp_img = Image.open(inp_path).resize((IMG_SIZE, IMG_SIZE))
    tar_img = Image.open(tar_path).resize((IMG_SIZE, IMG_SIZE))
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Hazy Image")
    plt.imshow(inp_img)
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.title("Clear Image (GT)")
    plt.imshow(tar_img)
    plt.axis("off")
    
    plt.show()


# ---------- 2. Noisy → Original ----------
print("Noisy → Original samples")
for i in range(num_samples):
    inp_path, tar_path = noisy_pairs[i]
    
    inp_img = Image.open(inp_path).resize((IMG_SIZE, IMG_SIZE))
    tar_img = Image.open(tar_path).resize((IMG_SIZE, IMG_SIZE))
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Noisy Image")
    plt.imshow(inp_img)
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.title("Original Image")
    plt.imshow(tar_img)
    plt.axis("off")
    
    plt.show()


# ---------- 3. Low-light → High-light ----------
print("Low-light → High-light samples")
for i in range(num_samples):
    inp_path, tar_path = lowlight_pairs[i]
    
    inp_img = Image.open(inp_path).resize((IMG_SIZE, IMG_SIZE))
    tar_img = Image.open(tar_path).resize((IMG_SIZE, IMG_SIZE))
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Low-light Image")
    plt.imshow(inp_img)
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.title("High-light Image")
    plt.imshow(tar_img)
    plt.axis("off")
    
    plt.show()

# %% Cell 6


  # or whatever size you want

# Example: define dataset directories


# List files inside each directory
hazy_files = sorted(os.listdir(hazy_dir))
clear_files = sorted(os.listdir(clear_dir))

# Pick the first file
img_hazy_path = os.path.join(hazy_dir, hazy_files[0])
img_clear_path = os.path.join(clear_dir, clear_files[0])

# Open image
img_hazy = Image.open(img_hazy_path).resize((IMG_SIZE, IMG_SIZE))
img_clear = Image.open(img_clear_path).resize((IMG_SIZE, IMG_SIZE))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Hazy Image")
plt.imshow(img_hazy)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Clear Image (GT)")
plt.imshow(img_clear)
plt.axis("off")

plt.show()

# %% Cell 7

def load_all_pairs(all_pairs, img_size=128):
    X, Y = [], []
    for pair in all_pairs:
        if len(pair) != 2:  # skip invalid entries
            continue
        inp_path, tar_path = pair
        inp = Image.open(inp_path).convert("RGB").resize((img_size, img_size))
        tar = Image.open(tar_path).convert("RGB").resize((img_size, img_size))
        
        X.append(np.array(inp) / 255.0)
        Y.append(np.array(tar) / 255.0)
    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Example: split dataset into train and test
train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)

X_train, y_train = load_all_pairs(train_pairs, img_size=128)
X_test, y_test = load_all_pairs(test_pairs, img_size=128)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# %% Cell 8
def conv3x3(out_channels):
    return keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=(1,1), padding="same")

def conv1x1(out_channels):
    return keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=(1,1), padding="same")

# %% Cell 9
class MDTA(layers.Layer):
    def __init__(self, channels, num_heads=8, window_size=16, **kwargs):
        super(MDTA, self).__init__(**kwargs)
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = window_size

        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)

        self.q_conv = models.Sequential([
            layers.Conv2D(channels, 1, padding="same"),
            layers.DepthwiseConv2D(3, padding="same")
        ])
        self.k_conv = models.Sequential([
            layers.Conv2D(channels, 1, padding="same"),
            layers.DepthwiseConv2D(3, padding="same")
        ])
        self.v_conv = models.Sequential([
            layers.Conv2D(channels, 1, padding="same"),
            layers.DepthwiseConv2D(3, padding="same")
        ])

        self.out_conv = layers.Conv2D(channels, 1, padding="same")

    def call(self, x):
        B = tf.shape(x)[0]
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        ws = self.window_size
        nh = self.num_heads
        hd = self.head_dim

        x = self.norm(x)
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        # Split into windows
        def window_partition(tensor):
            tensor = tf.reshape(
                tensor, [B, H // ws, ws, W // ws, ws, C]
            )
            tensor = tf.transpose(tensor, [0, 1, 3, 2, 4, 5])  # [B, H/ws, W/ws, ws, ws, C]
            tensor = tf.reshape(
                tensor, [-1, ws * ws, C]
            )  # [B*num_windows, ws*ws, C]
            return tensor

        q_windows = window_partition(q)
        k_windows = window_partition(k)
        v_windows = window_partition(v)

        # Split heads
        def split_heads(tensor):
            tensor = tf.reshape(tensor, [-1, ws * ws, nh, hd])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [B*nW, nh, ws*ws, hd]

        q = split_heads(q_windows)
        k = split_heads(k_windows)
        v = split_heads(v_windows)

        # Attention inside each window
        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn / tf.math.sqrt(tf.cast(hd, tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [-1, ws * ws, C])

        # Merge windows
        def window_reverse(tensor):
            tensor = tf.reshape(tensor, [B, H // ws, W // ws, ws, ws, C])
            tensor = tf.transpose(tensor, [0, 1, 3, 2, 4, 5])
            tensor = tf.reshape(tensor, [B, H, W, C])
            return tensor

        out = window_reverse(out)
        out = self.out_conv(out)
        return out + x

# %% Cell 10
class GDFN(layers.Layer):
    def __init__(self, channels, expansion=2, **kwargs):
        super(GDFN, self).__init__(**kwargs)
        self.channels = channels
        self.expanded_channels = channels * expansion 

        
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)

       
        self.path1 = models.Sequential([
            layers.Conv2D(self.expanded_channels, kernel_size=1, strides=(1,1), padding="same"),
            layers.DepthwiseConv2D(kernel_size=3, strides=(1,1), padding="same")
        ])

       
        self.path2 = models.Sequential([
            layers.Conv2D(self.expanded_channels, kernel_size=1, strides=(1,1), padding="same"),
            layers.DepthwiseConv2D(kernel_size=3, strides=(1,1), padding="same")
        ])

       
        self.out_conv = layers.Conv2D(channels, kernel_size=1, padding="same")

    def call(self, x):
       
        x_norm = self.norm(x)

       
        p1 = self.path1(x_norm)                
        p2 = self.path2(x_norm)                

        
        gated = p1 * tf.nn.sigmoid(p2)

        
        out = self.out_conv(gated)

        
        return out + x
    

# %% Cell 11
class TransformerBlock(layers.Layer):
    def __init__(self, channels, expansion=2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.mdta = MDTA(channels)
        self.gdfn = GDFN(channels, expansion)

    def call(self, x):
        x = self.mdta(x)
        x = self.gdfn(x)
        return x

# %% Cell 12
class PromptGenerationModule(layers.Layer):
    def __init__(self, in_channels, num_prompts=4):
        super().__init__()
        self.num_prompts = num_prompts
        self.conv = conv3x3(in_channels)            
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_prompts)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, x):
        feat = self.conv(x)                
        pooled = self.gap(feat)            
        weights = self.softmax(self.fc(pooled)) 

        weighted_prompts = []
        for i in range(self.num_prompts):
            w = tf.reshape(weights[:, i], [-1, 1, 1, 1])  
            weighted_prompts.append(w * feat)            

        prompt = tf.reduce_sum(tf.stack(weighted_prompts, axis=1), axis=1)  
        return prompt

# %% Cell 13
class PromptInteractionModule(layers.Layer):
    def __init__(self, in_channels):
        super().__init__()
        # TransformerBlock needs correct channels after concat → 2 * in_channels
        self.transformer = TransformerBlock(2 * in_channels)
        self.conv1 = conv1x1(in_channels)
        self.conv3 = conv3x3(in_channels)

    def call(self, feat, prompt):
        x = tf.concat([feat, prompt], axis=-1)   
        x = self.transformer(x)                 
        x = self.conv1(x)                        
        x = self.conv3(x)                      
        return x

# %% Cell 14
class PromptBlock(layers.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.pgm = PromptGenerationModule(in_channels)
        self.pim = PromptInteractionModule(in_channels)

    def call(self, x):
        prompt = self.pgm(x)       
        out = self.pim(x, prompt)  
        return out

# %% Cell 15
def downsample(x, out_ch):
   
    return layers.Conv2D(out_ch, kernel_size=3, strides=2, padding='same')(x)

def upsample(x, out_ch):
   
    x = layers.UpSampling2D(size=(2,2), interpolation='nearest')(x)
    x = layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='same')(x)
    return x

# %% Cell 16
def build_restormer_prompt(
        input_shape=(128, 128, 3),
        base_channels=48,
        num_blocks_lv1=1,
        num_blocks_lv2=1,
        num_blocks_lv3=1,
        num_blocks_lv4=1,
        num_prompts=4):
    

    inp = layers.Input(shape=input_shape)

  
    F0 = conv3x3(base_channels)(inp)  
    for _ in range(num_blocks_lv1):
        F1 = TransformerBlock(base_channels)(F0)

    x2 = downsample(F1, base_channels * 2)   # (h/2, w/2, 2c)
    for _ in range(num_blocks_lv2):
        F2 = TransformerBlock(base_channels * 2)(x2)

    x3 = downsample(F2, base_channels * 4)   # (h/4, w/4, 4c)
    for _ in range(num_blocks_lv3):
        F3 = TransformerBlock(base_channels * 4)(x3)

    x4 = downsample(F3, base_channels * 8)   # (h/8, w/8, 8c)
    for _ in range(num_blocks_lv4):
        F4 = TransformerBlock(base_channels * 8)(x4)

   
    F4_hat = PromptBlock(base_channels * 8)(F4)

   
    U0 = upsample(F4_hat, base_channels * 4)   # (h/4, w/4, 4c)    
    U0C = layers.Concatenate(axis=-1)([U0, F3])   
    x5 = conv1x1(base_channels * 4)(U0C)
    F5 = TransformerBlock(base_channels * 4)(x5)
    F5_hat = PromptBlock(base_channels * 4)(F5)

    U1 = upsample(F5_hat, base_channels * 2)      
    U1C = layers.Concatenate(axis=-1)([U1, F2])
    x6 = conv1x1(base_channels * 2)(U1C)
    F6 = TransformerBlock(base_channels * 2)(x6)
    F6_hat = PromptBlock(base_channels * 2)(F6)

    U2 = upsample(F6_hat, base_channels)          
    U2C = layers.Concatenate(axis=-1)([U2, F1])
    x7 = conv1x1(base_channels)(U2C)
    F7 = TransformerBlock(base_channels)(x7)

    # Output layer
    out = conv3x3(input_shape[-1])(F7)
    out = layers.Add()([out, inp])  

    model = models.Model(inputs=inp, outputs=out, name="Restormer_PromptNet")
    return model

# %% Cell 17
X_train = tf.image.resize(X_train, (128, 128))
y_train = tf.image.resize(y_train, (128, 128))

# %% Cell 18
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# %% Cell 19
# Shuffle, batch, and prefetch
train_ds = train_ds.shuffle(buffer_size=100).batch(8).prefetch(tf.data.AUTOTUNE)

# Check shapes
for x, y in train_ds.take(1):
    print(x.shape, y.shape)  # Should print: (1, 128, 128, 3) (1, 128, 128, 3)

# %% Cell 20
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(1).prefetch(tf.data.AUTOTUNE)

# %% Cell 21
batch_size = 8

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# %% Cell 22
model = build_restormer_prompt(
    input_shape=(128,128,3),
    base_channels=32,
    num_blocks_lv1=1,
    num_blocks_lv2=1,
    num_blocks_lv3=1,
    num_blocks_lv4=1
)

# %% Cell 23
for x, y in test_ds.take(1):
    print(x.shape, y.shape)

for hazy, clear in test_ds.take(1):
    restored = model(hazy, training=False)  # Model prediction
    print("Hazy shape:", hazy.shape)
    print("Restored shape:", restored.shape)

# %% Cell 24
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
tf.keras.backend.clear_session()

# %% Cell 25
#loss function

def psnr_metric(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)
    
model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='mse',              
        metrics=[psnr_metric]
    )

# %% Cell 26
history = model.fit(train_ds, epochs=30, verbose=1)

# %% Cell 27
test_loss = model.evaluate(test_ds, verbose=1)

# %% Cell 28
# Take one batch from test dataset
for x, y in test_ds:
    preds = model.predict(x)
    # preds = np.clip(preds, 0, 1)

    print(preds.shape)

    
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow(x[2])

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(y[2])

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(preds[2])
    plt.show()
   
