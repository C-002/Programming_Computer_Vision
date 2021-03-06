from PIL import Image

pil_im = Image.open('../data/empire.jpg')
print(pil_im.format, pil_im.size, pil_im.mode)
#pil_im.show()

pil_im = Image.open('../data/empire.jpg').convert('L')
print(pil_im.format, pil_im.size, pil_im.mode)
#pil_im.show()

pil_im_tn = pil_im.copy().thumbnail((128, 128))
#pil_im_tn.show()

box = (100, 100, 400, 400)
region = pil_im.crop(box)
region.show()

region = region.transpose(Image.ROTATE_180)
pil_im.paste(region, box)
pil_im.show()

out = pil_im.resize((128, 128))
out.show()
out = pil_im.rotate(45)
out.show()
