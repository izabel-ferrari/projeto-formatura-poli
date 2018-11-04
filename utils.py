import cv2

FACE_SIZE = 256

def print_done():
    print('Pronto.')

def validate_input_image(IMG_FILEPATH, IMG_EXTENSION):
    try:
        image = cv2.cvtColor(cv2.imread(IMG_FILEPATH), cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape

        # Verifica se a imagem está no formato JPG
        if IMG_EXTENSION not in ['.JPG', '.jpg']:
            print('A imagem de entrada deve estar no formato .jpg')
            return None

        # Verifica se a imagem tem tamanho mínimo de 256 x 256
        elif (height or width) < 256:
            print('A imagem de entrada deve ter pelo menos 256 x 256 px \n')
            return None

        return image
    except:
        raise

def resize_face(w, h, image_cropped):
    if (w != h):
        # print('Não temos um quadrado')
        # A menor dimensão será substituída pela maior dimensão
        if w < h:
            w = h
        else:
            h = w

    if (w == FACE_SIZE) and (h == FACE_SIZE):
            # print('Rosto no tamanho adequado')
            return image_cropped

    if (w < FACE_SIZE) and (h < FACE_SIZE):
        # print('Redimensionando para cima com INTER_CUBIC')
        return cv2.resize(image_cropped, (FACE_SIZE, FACE_SIZE), interpolation = cv2.INTER_CUBIC)

    if (w > FACE_SIZE) and (h > FACE_SIZE):
        # print('Redimensionando para baixo com INTER_AREA')
        return cv2.resize(image_cropped, (FACE_SIZE, FACE_SIZE), interpolation = cv2.INTER_AREA)

    return None
