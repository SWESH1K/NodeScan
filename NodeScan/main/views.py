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

# model = load_model(os.path.join('static', 'models', 'Abdominal_UNet.keras'), custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient})

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import zipfile
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.base import ContentFile

@csrf_exempt
def home(request):
    if request.method == 'POST':
        model_name = request.POST.get('model', 'Abdominal_UNet.keras')
        model_path = os.path.join('static', 'models', model_name)
        model = load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_coefficient': dice_coefficient})

        ct_images = []
        images = request.FILES.getlist('images')
        img_names = []
        for img in images:
            img_name = img.name
            img_names.append(img_name)
            img = Image.open(img)
            img = img.resize((256, 256))
            normalized_image = normalize_image(img)
            np_image = np.array(normalized_image).reshape((256, 256, 1))  # Reshape the image
            ct_images.append(np_image)
        
        # Add any additional processing logic here if needed
        predictions = model.predict(np.array(ct_images).reshape((len(ct_images), 256, 256, 1)))
        predicted_images = [prediction.reshape((256, 256)) for prediction in predictions]

        # Generate base64 strings for each image
        image_base64_list = []
        for np_image, predicted_image, img_name in zip(ct_images, predicted_images, img_names):
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
            image_base64_list.append(image_base64)

        # Store the base64 strings in the session
        request.session['image_base64_list'] = image_base64_list

        # Manual pagination logic
        page_size = 2
        page_number = 1
        total_images = len(image_base64_list)
        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size
        page_images = image_base64_list[start_index:end_index]

        # Calculate pagination details
        total_pages = (total_images + page_size - 1) // page_size
        has_previous = page_number > 1
        has_next = page_number < total_pages

        # Pass the base64 strings and pagination details to the template context
        context = {
            'page_images': page_images,
            'page_number': page_number,
            'total_pages': total_pages,
            'has_previous': has_previous,
            'has_next': has_next,
            'model_name': model_name,
            "redirect_to_upload": True,
        }
        return render(request, "home.html", context)
    elif request.GET.get('download') == 'true':
        # Handle download request
        image_base64_list = request.session.get('image_base64_list', [])
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for idx, image_base64 in enumerate(image_base64_list):
                image_data = base64.b64decode(image_base64)
                zip_file.writestr(f'image_{idx + 1}.png', image_data)
        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="processed_images.zip"'
        return response
    else:
        # Handle clear request
        if request.GET.get('clear') == 'true':
            request.session['image_base64_list'] = []
            
        # Retrieve the base64 strings from the session
        image_base64_list = request.session.get('image_base64_list', [])

        # Manual pagination logic
        page_size = 2
        page_number = int(request.GET.get('page', 1))
        total_images = len(image_base64_list)
        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size
        page_images = image_base64_list[start_index:end_index]

        # Calculate pagination details
        total_pages = (total_images + page_size - 1) // page_size
        has_previous = page_number > 1
        has_next = page_number < total_pages

        # Pass the base64 strings and pagination details to the template context
        context = {
            'page_images': page_images,
            'page_number': page_number,
            'total_pages': total_pages,
            'has_previous': has_previous,
            'has_next': has_next,
            'model_name': request.GET.get('model', 'Abdominal_UNet.keras'),
            "redirect_to_upload": True if request.GET else False,
        }
        return render(request, "home.html", context)