from PIL import Image

# Open the image
img = Image.open("LaTex_Project/polimi_template_classic/Images/logo_polimi.png").convert("RGBA")

# Get the image data
data = list(img.getdata())

# Replace all black pixels (0, 0, 0) with white (255, 255, 255)
new_data = [
    (255, 255, 255, pixel[3]) if pixel[:3] == (0, 0, 0) else pixel
    for pixel in data
]
img.putdata(new_data)

# Save as PNG
img.save("mono_white_logo.png", format="PNG")
