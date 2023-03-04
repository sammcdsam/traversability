from models.led import *

def LEDnet(classes):
    inpu = Input(shape=(1208,1920,3))
    encode = encoder(inpu)
    dec = decoder(encode,classes)
    comp = Model(inputs=inpu,outputs=dec)
    return comp
