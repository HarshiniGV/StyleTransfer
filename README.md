# StyleTransfer
Neural Style Transfer (TensorFlow) — README

This project applies Neural Style Transfer using TensorFlow + VGG19.
It blends the content of one image with the style of another by optimizing a generated image using style/content feature losses.

Usage

Put your images in the project:

content_path = '/content/image1.jpg'
style_path = '/content/image2.jpg'


Run the script.
It:

Loads images

Extracts VGG19 features

Computes style/content loss

Optimizes the image for 25 epochs × 150 steps

Displays the final stylized output

Main Components

load_img() – loads & resizes image

StyleContentModel – extracts VGG19 style + content features

gram_matrix() – represents style

train_step() – performs Adam optimization

Final image is clipped to valid pixel range

Results

The generated output preserves:

Shapes & structure from the content image

Colors, textures, brush patterns from the style image

Higher epochs = smoother + more detailed style

Style weight ↑ → more painterly effect

Content weight ↑ → more original structure retaine
