import cv2
import math
import numpy as np

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
PI = 3.14
WAIT_TIME = 1000
HOLES = [(240, 136),
         (942, 136),
         (1684, 136),
         (1684, 802),
         (942, 814),
         (240, 798)]
LINE_THICKNESS = 4
GHOST_SIZE = 25


def pre_processing(frame, threshold1, threshold2, k=3):
    frame_processed = cv2.GaussianBlur(frame, (5, 5), 3)
    frame_processed = cv2.Canny(frame_processed, threshold1, threshold2)
    kernel = np.ones((k, k), np.uint8)
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


def calculate_projection(x1, y1, x2, y2, length):
    length_AB = ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x3 = int(x2 + (x2 - x1) / int(length_AB) * length)
    y3 = int(y2 + (y2 - y1) / int(length_AB) * length)

    return x3, y3


def draw_contour(img, x, y, w, h, i=100):
    cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
    cv2.circle(img, (x + (w // 2), y + (h // 2)), 5, RED, cv2.FILLED)
    cv2.circle(img, (x + (w // 2), y + (h // 2)), 25, RED, 1)
    if i != 100:
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)


# Check if any hole is hit by calculating if is
# inside one another
def result(frame, x, y, radius=35):
    inside = 0
    for hole in HOLES:
        square_dist = (hole[0] - x) ** 2 + (hole[1] - y) ** 2
        if square_dist < radius ** 2:
            inside += 1

    if inside > 0:
        cv2.putText(frame, 'IN', (750, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, GREEN, 9)
        return GREEN
    else:
        cv2.putText(frame, 'OUT', (750, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, RED, 9)
        return RED


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


def find_angle(img, point1, point2, point3, show_degrees=False):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    x3, y3 = point3[0], point3[1]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if show_degrees:
        cv2.putText(img, str(int(angle + 360)), (x2 - 50, y2 + 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    return angle + 360 if angle < 0 else angle
