import os
from PIL import Image, ImageDraw

def generate_logo():
    # Make sure assets folder exists
    os.makedirs("assets", exist_ok=True)
    
    # Dimensions
    width, height = 512, 512
    
    # Create image with deep slate/indigo background
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255)) # #0F172A
    draw = ImageDraw.Draw(img)
    
    # Draw geometric glowing background circle
    draw.ellipse([64, 64, 448, 448], fill=(79, 70, 225, 40), outline=(79, 70, 225, 255), width=4)
    
    # Draw a smaller nested accent circle
    draw.ellipse([96, 96, 416, 416], fill=None, outline=(52, 211, 153, 100), width=2)
    
    # Draw a central stylized neural node/book emblem
    # Draw central vertical spine
    draw.line([256, 160, 256, 352], fill=(248, 250, 252, 255), width=8)
    
    # Draw Left Page
    draw.polygon([256, 176, 176, 208, 176, 320, 256, 288], fill=(79, 70, 225, 200), outline=(248, 250, 252, 255), width=4)
    
    # Draw Right Page
    draw.polygon([256, 176, 336, 208, 336, 320, 256, 288], fill=(52, 211, 153, 200), outline=(248, 250, 252, 255), width=4)
    
    # Draw neural node links
    # Top left node
    draw.ellipse([160, 192, 192, 224], fill=(52, 211, 153, 255))
    # Top right node
    draw.ellipse([320, 192, 352, 224], fill=(79, 70, 225, 255))
    # Center top node
    draw.ellipse([240, 144, 272, 176], fill=(248, 250, 252, 255))
    
    # Save the logo
    img.save("assets/logo.png", "PNG")
    print("Logo successfully generated at assets/logo.png")

if __name__ == "__main__":
    generate_logo()
