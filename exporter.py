import cv2 as cv
class chess_exporter():

    def __int__(self, path="out/"):
        self.path = path


    def image(self, image, name)->None:
        cv.imwrite(filename=f'{self.path}{name}', img=image)

