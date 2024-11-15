from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os
import numpy as np
from django.http import HttpResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from PIL import Image

# Load the model 

# Define the Dice coefficient metric
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Define the Dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Define the combined loss function
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def normalize_image(image: Image.Image) -> Image.Image:
    """
    Normalize the image to the range 0-255.
    :param image: The image to normalize.
    :return: A normalized image.
    """
    np_image = np.array(image)
    min_val = np.min(np_image)
    max_val = np.max(np_image)

    # Normalize to 0-255
    normalized_image = ((np_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return Image.fromarray(normalized_image)

model = load_model('static\models\Abdominal_UNet.keras', custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient})

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from django.shortcuts import render
from django.http import HttpResponse

def home(request):
    if request.method == 'POST':
        ct_images = []
        img = request.FILES.get('image')
        if img:
            img_name = img.name 
            img = Image.open(img)
            img = img.resize((256, 256))
            normalized_image = normalize_image(img)
            np_image = np.array(normalized_image).reshape((256, 256, 1))  # Reshape the image
            ct_images.append(np_image)
            # Add any additional processing logic here if needed
            prediction = model.predict(np.array(ct_images).reshape((1, 256, 256, 1)))
            predicted_image = prediction[0].reshape((256, 256))  # Reshape the predicted image

            # Plot the original and predicted images
            plt.figure(figsize=(12, 4))

            # Original image
            plt.subplot(1, 2, 1)
            plt.imshow(np_image.squeeze(), cmap='gray')  # Remove channel dimension for display
            plt.title("Original Image")
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 2, 2)
            plt.imshow(predicted_image, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')

            # Add label in the bottom right corner
            plt.text(1, 0, "Scanned using NodeScan  ", ha='right', va='bottom', transform=plt.gcf().transFigure, fontsize=12)

            # Add image name in the bottom left corner
            plt.text(0, 0, f"  {img_name}", ha='left', va='bottom', transform=plt.gcf().transFigure, fontsize=12, color='#E0FBFC')

            # Set the background color to match the website color palette
            plt.gcf().patch.set_facecolor('#5C6B73')

            # Save the plot to a BytesIO buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Convert the buffer to a base64 string
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Pass the base64 string to the template context
            context = {
                'image_base64': image_base64,
                "redirect_to_upload": True,
            }
            return render(request, "home.html", context)
        else:
            print("No Image")
    return render(request, "home.html")