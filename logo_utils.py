from PIL import Image

def detect_logo(yolo, img_path):
    try:
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except:
        print('File Open Error! Try again!')
        return None, None

    prediction, new_image = yolo.detect_image(image)

    return prediction, new_image
