import pytesseract
from PIL import Image
im = Image.open('./Untitled.png')
#print(pytesseract.image_to_string(im, lang='chi_sim'))
code = pytesseract.image_to_string(im, lang='chi_sim')
print(code)
print('植物')