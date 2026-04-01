import cv2
import numpy as np

def augment_image(image):
    """
    Realiza o data augmentation em uma imagem.
    Aplica as seguintes transformações:
    - Inversão horizontal
    - Rotação de 15 graus
    - Blur (desfoque)

    Args:
        image (numpy.ndarray): Imagem original.

    Returns:
        dict: Dicionário contendo as três imagens com suas respectivas transformações.
    """
    # 1. Inversão horizontal
    flipped_image = cv2.flip(image, 1)
    
    # 2. Diferença de ângulo em 15 graus
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    
    # 3. Blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return {
        "flipped": flipped_image,
        "rotated": rotated_image,
        "blurred": blurred_image
    }

def apply_sequential_augmentation(image):
    """
    Aplica as transformações de forma sequencial na mesma imagem.
    
    Args:
        image (numpy.ndarray): Imagem original.
        
    Returns:
        numpy.ndarray: Imagem modificada com todas as transformações aplicadas juntas.
    """
    aug_img = cv2.flip(image, 1)
    
    (h, w) = aug_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    aug_img = cv2.warpAffine(aug_img, M, (w, h))
    
    # Blur
    aug_img = cv2.GaussianBlur(aug_img, (5, 5), 0)
    
    return aug_img
