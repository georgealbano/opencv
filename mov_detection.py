import cv2
import numpy as np
URL_VIDEO = 'http://45.171.207.7:4333/hls/1_205.m3u8'
BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
BGS_TYPE = BGS_TYPES[2]


def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == 'opening':
        return np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        return np.ones((3, 3), np.uint8)


def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, getKernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, getKernel('closing'), iterations=2)
        opening = cv2.morphologyEx(
            closing, cv2.MORPH_OPEN, getKernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, getKernel('dilation'), iterations=2)
        return dilation


def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120)
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(500, 8, 0.1, 0)
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    raise ValueError('Invalid background subtraction type: ' + BGS_TYPE)


bg_subtractor = getBGSubtractor(BGS_TYPE)


def main():
    cap = cv2.VideoCapture(URL_VIDEO)
    while (cap.isOpened):
        _, frame = cap.read()

        if not _:
            print('ERRo')
            return

        bg_mask = bg_subtractor.apply(frame)
        bg_mask = getFilter(bg_mask, 'combine')
        bg_mask = cv2.medianBlur(bg_mask, 5)

        countors, hierarchy = cv2.findContours(
            bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in countors:
            area = cv2.contourArea(cnt)
            if area > 250:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (10, 30), (300, 55), (255, 0, 0), -1)
                cv2.putText(frame, 'Movimento detectado!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawContours(frame, cnt, 1, (0, 255, 0), 10)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)

        # result = cv2.bitwise_and(frame, frame)
        cv2.imshow('Video', frame)
        cv2.imshow('mask', bg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
