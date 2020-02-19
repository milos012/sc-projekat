import cv2
import numpy as np
import imutils
from sklearn.metrics import mean_absolute_error


def presao(x, y ):
    eq = y - 250
    if abs(eq) <= 1:
        return True
    return False


def obradiVideo(path):
    video = cv2.VideoCapture(path)
    pocetak = None
    prebrojanoLjudi = 0

    while (video.isOpened()):
        (bool, frame) = video.read()
        if bool != True:
            break

        frame = imutils.resize(frame, width=810)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (21, 21), 0)

        if pocetak is None:
            pocetak = img
            continue

        distance = cv2.absdiff(pocetak, img)
        sredjena = cv2.threshold(distance, 21, 255, cv2.THRESH_BINARY)[1]
        # less_noice = cv2.erode(thresh, (3, 3), iterations=4)
        sredjena = cv2.dilate(sredjena, None, iterations=4)

        konture, _ = cv2.findContours(sredjena,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        for kontura in konture:
            if cv2.contourArea(kontura) < 120:
                continue

            #dimenzije kontura
            (x, y, w, h) = cv2.boundingRect(kontura)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(frame, (800 // 4, 150), (600, 120), (0, 255, 0), 2)
            centarX = (x + x + w) // 2
            centarY = (y + y + h) // 2
            centarKonture = (centarX,  centarY)

            cv2.circle(frame, centarKonture, 1, (255, 0, 0), 3)

            if (presao(centarX, centarY)):
                prebrojanoLjudi = prebrojanoLjudi + 1


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("Application", frame)
    return prebrojanoLjudi

if __name__ == "__main__":
    video1 = "snimci/video1.mp4"
    video2 = "snimci/video2.mp4"
    video3 = "snimci/video3.mp4"
    video4 = "snimci/video4.mp4"
    video5 = "snimci/video5.mp4"
    video6 = "snimci/video6.mp4"
    video7 = "snimci/video7.mp4"
    video8 = "snimci/video8.mp4"
    video9 = "snimci/video9.mp4"
    video10 = "snimci/video10.mp4"
    
    resenja = []
    resenja.append(obradiVideo(video1))
    resenja.append(obradiVideo(video2))
    resenja.append(obradiVideo(video3))
    resenja.append(obradiVideo(video4))
    resenja.append(obradiVideo(video5))
    resenja.append(obradiVideo(video6))
    resenja.append(obradiVideo(video7))
    resenja.append(obradiVideo(video8))
    resenja.append(obradiVideo(video9))
    resenja.append(obradiVideo(video10))
    print("Dobijena resenja:")
    for r in resenja:
        print(r)
    tacna = [4,24,17,23,17,27,29,22,10,23]
    print("Tacna resenja: 4,24,17,23,17,27,29,22,10,23")
    print("Greska:"+  str(mean_absolute_error(tacna,resenja)))
