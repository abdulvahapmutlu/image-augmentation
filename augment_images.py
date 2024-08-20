import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import save_img

# Define the augmentations
def augment_image(image):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert image to float32 for augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    image = tf.add(image, noise)
    
    image = tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05)(image)
    
    image = tf.clip_by_value(image, 0.0, 1.0)  # Ensure pixel values are in the range [0, 1]
    image = tf.image.convert_image_dtype(image, tf.uint8)  # Convert back to uint8 for saving
    return image

# Load and augment the dataset
def load_and_augment_images(dataset_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    for filename in os.listdir(dataset_path):
        if os.path.splitext(filename)[-1].lower() in image_extensions:
            img_path = os.path.join(dataset_path, filename)
            image = tf.image.decode_image(tf.io.read_file(img_path), channels=3)
            
            # Apply and save augmentations
            for i in range(6):  # Apply 6 augmentations to create 7 variations in total
                augmented_image = augment_image(image)
                new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[-1]}"
                
                # Save the augmented image with the same format and quality as the original
                save_img(os.path.join(dataset_path, new_filename), augmented_image.numpy(), quality=100)

dataset_path = "path/to/dataset"
load_and_augment_images(dataset_path)
