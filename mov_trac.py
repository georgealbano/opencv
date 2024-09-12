from random import randint
import cv2

URL_VIDEO = '/home/knet/George/pythonProject/Curso_Subtraction/videos/Pedestrians_2.mp4'


# listas de algoritimos de rastreamento
tracker_types = ['MIL', 'KCF', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL.create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF.create()
if tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT.create()


cap = cv2.VideoCapture(URL_VIDEO)
print(tracker)
_, frame = cap.read()
frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
bbox = cv2.selectROI(frame, False)


# verificar o retorno
ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))


while (cap.isOpened):
    _, frame = cap.read()

    if not _:
        print('ERRo')
        break

    frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
    timer = cv2.getTickCount()
    ok, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        cv2.putText(frame, 'Falha no rastreamento ', (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.putText(frame, tracker_type + ' Tracker', (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.putText(frame, 'fps' + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
