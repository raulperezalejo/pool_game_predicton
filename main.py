import cv2
import numpy as np
import cvzone
import datetime
from pool_utils import *
import math
import time

FIRST_SHOOT = 90

cap = cv2.VideoCapture('video.mp4')

# def empty(a):
#     pass

# # Trackball to adjuts Canny
# cv2.namedWindow('Settings')
# cv2.resizeWindow('Settings', 640, 240)
# cv2.createTrackbar('Threshold1', 'Settings', 50, 255, empty)
# cv2.createTrackbar('Threshold2', 'Settings', 50, 255, empty)


# # count the number of frames
# frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# # calculate duration of the video
# seconds = round(frames / fps)
# video_time = datetime.timedelta(seconds=seconds)
#
# print(f"frames: {frames}")
# print(f"fps: {fps}")
# print(f"duration in seconds: {seconds}")
# print(f"video time: {video_time}")


frame_count = 0
while cap.isOpened():
    frame_count += 1
    success, frame = cap.read()

    frame_processed = pre_processing(frame)

    frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=1000)

    # frame_stacked = cvzone.stackImages([frame_processed], 1, 2)

    if frame_count == 44:
        # Contour filtering
        cnt_found = [d for d in cnt_found if 300 < d['bbox'][0] < 600 and d['bbox'][1] < 800]

        while True:
            key2 = cv2.waitKey(1) or 0xff

            # Stick
            xs, ys, w, h = cnt_found[0]['bbox']
            xs_center, ys_center = xs + (w // 2), ys + (h // 2)
            # Cue
            xc, yc, w, h = cnt_found[1]['bbox']
            xc_center, yc_center = xc + (w // 2), yc + (h // 2)
            # ghost ball trajectory
            x3, y3 = calculate_projection(frame_countours, xs_center, ys_center, xc_center, yc_center, 7)

            # Ball
            xb, yb, w, h = cnt_found[3]['bbox']
            xb_center, yb_center = xb + (w // 2), yb + (h // 2)
            # Ball trajectory
            calculate_projection(frame_countours, x3, y3, xb_center, yb_center, 10)

            text_result(frame_countours, 'IN', GREEN)

            cv2.imshow('frame', frame_countours)

            if key2 == ord('c'):
                break

    ######################

    if frame_count == 100:
        # border line
        b_x1, b_y1, b_x2, b_y2 = 500, 800, 900, 800
        cv2.line(frame_countours, (b_x1, b_y1), (b_x2, b_y2), RED, 4)

        cnt_found = [d for d in cnt_found if 300 < d['bbox'][0] < 800 and d['bbox'][1] < 800]

        while True:
            key2 = cv2.waitKey(1) or 0xff
            # frame_processed = pre_processing(frame)
            # cv2.circle(frame_countours, (229, 135), 35, BLACK, cv2.FILLED)
            # cv2.line(frame_countours, (521, 710), (475, 498), RED, 4)
            #
            # print(len(cnt_found))
            # for cnt in cnt_found:
            #     x, y, w, h = cnt['bbox']
            #     draw_contour(frame_countours, x, y, w, h)
            # xs, ys, w, h = cnt_found[2]['bbox']
            # draw_contour(frame_countours, xs, ys, w, h)

            # Stick
            xs, ys, w, h = cnt_found[3]['bbox']
            draw_contour(frame_countours, xs, ys, w, h)
            xs_center, ys_center = xs + (w // 2), ys + (h // 2)

            # # cue
            xc, yc, w, h = cnt_found[1]['bbox']
            draw_contour(frame_countours, xc, yc, w, h)
            xc_center, yc_center = xc + (w // 2), yc + (h // 2)
            #
            # ghost ball trajectory
            x3, y3 = calculate_projection(frame_countours, xs_center, ys_center, xc_center, yc_center, 10)

            # ball
            xb, yb, w, h = cnt_found[2]['bbox']
            draw_contour(frame_countours, xb, yb, w, h)

            xb_center, yb_center = xb + (w // 2), yb + (h // 2)

            # number 10 ball trajectory
            b_projection_x, b_projection_y = calculate_projection(frame_countours, x3, y3, xb_center, yb_center, 30)

            intersection_point = line_intersection(((b_x1, b_y1), (b_x2, b_y2)),
                                                   ((xb_center, yb_center), (b_projection_x, b_projection_y)))

            cv2.circle(frame_countours, (intersection_point[0], intersection_point[1]), 10, BLACK, cv2.FILLED)

            ang = angle(frame_countours, (b_x1, b_y1), intersection_point, (xb_center, yb_center))

            ang = -110
            l = 120
            # # apx, apy = intersection_point[0] + l * math.cos(ang), \
            # #            intersection_point[1] + l * math.sin(ang)
            x2 = intersection_point[0] + (l * math.sin(ang))
            y2 = intersection_point[1] + (l * math.cos(ang))
            # # print(int(apx), int(apy))
            cv2.line(frame_countours, (intersection_point[0], intersection_point[1]),
                     (int(x2), int(y2)), WHITE, 4)

            # text_result(frame_countours, 'IN', GREEN)

            cv2.imshow('frame', frame_countours)

            if key2 == ord('c'):
                break

    ######################

    if frame_count == 274:
        cv2.imwrite("frame1.jpg", frame)  # save frame as JPEG file

        # border line
        b_x1, b_y1, b_x2, b_y2 = 400, 200, 800, 800
        cv2.rectangle(frame_countours, (b_x1, b_y1), (b_x2, b_y2), RED, 4)

        cnt_found = [d for d in cnt_found if 400 < d['bbox'][0] < 800 and 200 < d['bbox'][1] < 800]

        while True:
            key2 = cv2.waitKey(1) or 0xff
            # frame_processed = pre_processing(frame)
            # cv2.circle(frame_countours, (229, 135), 35, BLACK, cv2.FILLED)
            # cv2.line(frame_countours, (521, 710), (475, 498), RED, 4)
            #
            print(len(cnt_found))
            for cnt in cnt_found:
                x, y, w, h = cnt['bbox']
                draw_contour(frame_countours, x, y, w, h)
            # xs, ys, w, h = cnt_found[2]['bbox']
            # draw_contour(frame_countours, xs, ys, w, h)

            # # Stick
            # xs, ys, w, h = cnt_found[3]['bbox']
            # draw_contour(frame_countours, xs, ys, w, h)
            # xs_center, ys_center = xs + (w // 2), ys + (h // 2)


            cv2.imshow('frame', frame_countours)

            if key2 == ord('c'):
                break


    cv2.imshow('frame', frame_countours)

    key = cv2.waitKey(1)
    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            frame_processed = pre_processing(frame)
            cv2.imshow('frame', frame_countours)

            if key2 == ord('p'):
                break
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
