import cv2
import math
import numpy as np

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


def pre_processing(frame):
    frame_processed = cv2.GaussianBlur(frame, (5, 5), 3)
    # threshold_1 = cv2.getTrackbarPos('Threshold1', 'Settings')
    # threshold_2 = cv2.getTrackbarPos('Threshold2', 'Settings')
    # frame_processed = cv2.Canny(frame_processed, threshold_1, threshold_2)
    frame_processed = cv2.Canny(frame_processed, 96, 75)
    kernel = np.ones((3, 3), np.uint8)
    frame_processed = cv2.dilate(frame_processed, kernel, iterations=1)
    frame_processed = cv2.morphologyEx(frame_processed, cv2.MORPH_CLOSE, kernel)
    return frame_processed


def find_contours(img, imgPre, minArea=1000, sort=True, filter=0, drawCon=False, c=(0, 255, 0)):
    """
    Finds Contours in an image
    :param img: Image on which we want to draw
    :param imgPre: Image on which we want to find contours
    :param minArea: Minimum Area to detect as valid contour
    :param sort: True will sort the contours by area (biggest first)
    :param filter: Filters based on the corner points e.g. 4 = Rectangle or square
    :param drawCon: draw contours boolean
    :return: Foudn contours with [contours, Area, BoundingBox, Center]
    """
    conFound = []
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgPre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            if len(approx) == filter or filter == 0:
                x, y, w, h = cv2.boundingRect(approx)
                cx, cy = x + (w // 2), y + (h // 2)
                if drawCon:
                    cv2.drawContours(imgContours, cnt, -1, c, 3)
                    cv2.rectangle(imgContours, (x, y), (x + w, y + h), c, 2)
                    cv2.circle(imgContours, (x + (w // 2), y + (h // 2)), 5, c, cv2.FILLED)
                    cv2.putText(imgContours, str(area), (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 2)
                conFound.append({"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": [cx, cy]})

    if sort:
        conFound = sorted(conFound, key=lambda x: x["area"], reverse=True)

    return imgContours, conFound


def click_event(event, x, y, _flags, _params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)


def calculate_projection(img, x1, y1, x2, y2, length):
    length_AB = math.sqrt((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
    x3 = int(x2 + (x2 - x1) / int(length_AB) * length)
    y3 = int(y2 + (y2 - y1) / int(length_AB) * length)

    cv2.line(img, (x2, y2), (x3, y3), YELLOW, 4)
    cv2.circle(img, (x3, y3), 25, WHITE, cv2.FILLED)
    return x3, y3


def draw_contour(img, x, y, w, h):
    # print(x, y, w, h)
    # cv2.drawContours(frame_countours, cnt['cnt'], -1, c, 3)
    cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
    cv2.circle(img, (x + (w // 2), y + (h // 2)), 5, RED, cv2.FILLED)
    cv2.circle(img, (x + (w // 2), y + (h // 2)), 25, RED, 1)
    cv2.putText(img, str(y), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)


def text_result(img, txt, color):
    cv2.putText(img, txt, (700, 400), cv2.FONT_HERSHEY_SIMPLEX, 7, color, 9)


def line_intersection(line1, line2):
    """
    https://stackoverflow.com/a/20677983
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def angle(img, point1, point2, point3):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    x3, y3 = point3[0], point3[1]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.line(img, (x3, y3), (x2, y2), (255, 0, 255), 3)
    cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
    cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
    cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
    cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if angle < 0:
        angle += 360

    return angle
