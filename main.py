from pool_utils import *

cap = cv2.VideoCapture('video.mp4')

# Bottom edge bounce coordinates
edge_x1, edge_y1, edge_x2, edge_y2 = 270, 800, 1650, 800


frame_count = 0
while cap.isOpened():
    frame_count += 1
    success, frame = cap.read()

    # Holes will always be visible
    for i in HOLES:
        cv2.circle(frame, i, 35, WHITE, cv2.FILLED)

    if frame_count == 44:
        frame_processed = pre_processing(frame, threshold1=96, threshold2=75)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=1000)

        # Contour filtering
        f_x1, f_x2, f_y2 = 300, 600, 800
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and d['bbox'][1] < f_y2]

        # Stick
        xs, ys, w, h = cnt_found[0]['bbox']
        xs_center, ys_center = xs + (w // 2), ys + (h // 2)

        # Cue
        xc, yc, w, h = cnt_found[1]['bbox']
        xc_center, yc_center = xc + (w // 2), yc + (h // 2)

        # Ball
        xb, yb, w, h = cnt_found[3]['bbox']
        xb_center, yb_center = xb + (w // 2), yb + (h // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 35500)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 13000)

        color = result(frame, x_cb, y_cb)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw ball trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)
        cv2.circle(frame, (x_cb, y_cb), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 100:
        frame_processed = pre_processing(frame, 96, 75)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=1000)

        # Bottom edge line
        cv2.line(frame, (edge_x1, edge_y1), (edge_x2, edge_y2), YELLOW, 4)

        # Contour filtering
        f_x1, f_x2, f_y2 = 300, 800, 800
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and d['bbox'][1] < f_y2]

        # Stick
        xs, ys, w, h = cnt_found[3]['bbox']
        xs_center, ys_center = xs + (w // 2), ys + (h // 2)

        # Cue
        xc, yc, w, h = cnt_found[1]['bbox']
        xc_center, yc_center = xc + (w // 2), yc + (h // 2)

        # Ball
        xb, yb, w, h = cnt_found[2]['bbox']
        xb_center, yb_center = xb + (w // 2), yb + (h // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 20000)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 8000)

        # Calculating intersection point and angle with bottom edge
        intersection_point = line_intersection(((edge_x1, edge_y1), (edge_x2, edge_y2)),
                                               ((xb_center, yb_center), (x_cb, y_cb)))

        angle = find_angle(frame, (edge_x1, edge_y1), intersection_point, (xb_center, yb_center))
        x_i = int(intersection_point[0] + (700 * math.cos(-angle * PI / 180.0)))
        y_i = int(intersection_point[1] + (700 * math.sin(-angle * PI / 180.0)))

        color = result(frame, x_i, y_i)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw bounce trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)

        cv2.line(frame, (intersection_point[0], intersection_point[1]), (x_i, y_i), color, LINE_THICKNESS)
        cv2.circle(frame, (x_i, y_i), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 274:
        frame_processed = pre_processing(frame, 30, 30)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=500)

        # Contour filtering
        f_x1, f_y1, f_x2, f_y2 = 400, 200, 800, 800
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and f_y1 < d['bbox'][1] < f_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[1]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[2]['bbox']
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[4]['bbox']
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 23500)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 26000)

        color = result(frame, x_cb, y_cb)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw ball trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)
        cv2.circle(frame, (x_cb, y_cb), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 369:
        frame_processed = pre_processing(frame, 38, 81)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=500)

        # Bottom edge line
        cv2.line(frame, (edge_x1, edge_y1), (edge_x2, edge_y2), YELLOW, 4)

        # Contour filtering
        f_x1, f_y1, f_x2, f_y2 = 400, 000, 800, 800
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and f_y1 < d['bbox'][1] < f_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[7]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[4]['bbox']
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[2]['bbox']
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 32000)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 13500)

        intersection_point = line_intersection(((edge_x1, edge_y1), (edge_x2, edge_y2)),
                                               ((xb_center, yb_center), (x_cb, y_cb)))

        angle = find_angle(frame, (edge_x1, edge_y1), intersection_point, (xb_center, yb_center))
        x_i = intersection_point[0] + (700 * math.cos(-angle * PI / 180.0))
        y_i = intersection_point[1] + (700 * math.sin(-angle * PI / 180.0))

        color = result(frame, x_i, y_i)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw bounce trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)

        cv2.line(frame, (intersection_point[0], intersection_point[1]), (int(x_i), int(y_i)), color, LINE_THICKNESS)
        cv2.circle(frame, (int(x_i), int(y_i)), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 504:
        frame_processed = pre_processing(frame, 96, 75)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=500)

        # Contour filtering
        f_x1, f_y1, f_x2, f_y2 = 1100, 50, 1800, 800
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and f_y1 < d['bbox'][1] < f_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[3]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[8]['bbox']  # 7
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[9]['bbox']  # 4
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 54500)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 10000)

        color = result(frame, x_cb, y_cb)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw ball trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)
        cv2.circle(frame, (x_cb, y_cb), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################
    if frame_count == 644:
        frame_processed = pre_processing(frame, 51, 156)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=500)

        # Contour filtering
        f_x1, f_y1, f_x2, f_y2 = 200, 250, 800, 800
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and f_y1 < d['bbox'][1] < f_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[2]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[3]['bbox']
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[4]['bbox']
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 36000)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 10000)

        color = result(frame, x_cb, y_cb)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw ball trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)
        cv2.circle(frame, (x_cb, y_cb), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 762:
        frame_processed = pre_processing(frame, 53, 108, 4)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=500)

        # Contour filtering
        f_x1, f_y1, f_x2, f_y2 = 800, 0, 1400, 600
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and f_y1 < d['bbox'][1] < f_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[2]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[3]['bbox']  # 7
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[6]['bbox']  # 4
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 65000)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 10000)

        color = result(frame, x_cb, y_cb)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw ball trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)
        cv2.circle(frame, (x_cb, y_cb), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 908:
        frame_processed = pre_processing(frame, 88, 132, 5)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=100)

        # Bottom edge line
        cv2.line(frame, (edge_x1, edge_y1), (edge_x2, edge_y2), YELLOW, 4)

        # Contour filtering
        f_x1, f_y1, f_x2, f_y2 = 400, 0, 1100, 750
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and f_y1 < d['bbox'][1] < f_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[9]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[2]['bbox']
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[1]['bbox']
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 53000)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 4700)

        intersection_point = line_intersection(((edge_x1, edge_y1), (edge_x2, edge_y2)),
                                               ((xb_center, yb_center), (x_cb, y_cb)))

        angle = find_angle(frame, (edge_x1, edge_y1), intersection_point, (xb_center, yb_center))
        x_i = intersection_point[0] + (800 * math.cos(-angle * PI / 180.0))
        y_i = intersection_point[1] + (800 * math.sin(-angle * PI / 180.0))

        color = result(frame, x_i, y_i)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw bounce trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)

        cv2.line(frame, (intersection_point[0], intersection_point[1]), (int(x_i), int(y_i)), color, LINE_THICKNESS)
        cv2.circle(frame, (int(x_i), int(y_i)), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 1061:
        frame_processed = pre_processing(frame, 61, 207, 4)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=500)

        # Contour filtering
        f_x1, f_y1, f_x2, f_y2 = 400, 0, 1100, 800
        cnt_found = [d for d in cnt_found if f_x1 < d['bbox'][0] < f_x2 and f_y1 < d['bbox'][1] < f_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[4]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[5]['bbox']
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[2]['bbox']
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center, ys_center, xc_center, yc_center, 9200)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 33000)

        color = result(frame, x_cb, y_cb)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw ball trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)
        cv2.circle(frame, (x_cb, y_cb), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    ######################

    if frame_count == 1163:
        frame_processed = pre_processing(frame, 73, 85, 4)
        frame_countours, cnt_found = find_contours(frame, frame_processed, minArea=500)

        # Bottom edge line
        cv2.line(frame, (edge_x1, edge_y1), (edge_x2, edge_y2), YELLOW, 4)

        # Contour filtering
        b_x1, b_y1, b_x2, b_y2 = 1200, 0, 1700, 650
        cnt_found = [d for d in cnt_found if b_x1 < d['bbox'][0] < b_x2 and b_y1 < d['bbox'][1] < b_y2]

        # Stick
        xs, ys, ws, hs = cnt_found[1]['bbox']
        xs_center, ys_center = xs + (ws // 2), ys + (hs // 2)

        # Cue
        xc, yc, wc, hc = cnt_found[6]['bbox']
        xc_center, yc_center = xc + (wc // 2), yc + (hc // 2)

        # Ball
        xb, yb, wb, hb = cnt_found[8]['bbox']
        xb_center, yb_center = xb + (wb // 2), yb + (hb // 2)

        # In this particular case the ball is not hit in the middle
        # For this reason, this Offset in added.
        OFFSET = 20

        # Calculating ghost ball trajectory
        x_sc, y_sc = calculate_projection(xs_center+OFFSET, ys_center, xc_center, yc_center, 23000)

        # Calculating ball trajectory
        x_cb, y_cb = calculate_projection(x_sc, y_sc, xb_center, yb_center, 12000)

        intersection_point = line_intersection(((edge_x1, edge_y1), (edge_x2, edge_y2)),
                                               ((xb_center, yb_center), (x_cb, y_cb)))

        angle = find_angle(frame, (edge_x1, edge_y1), intersection_point, (xb_center, yb_center))
        x_i = intersection_point[0] + (700 * math.cos(-angle * PI / 180.0))
        y_i = intersection_point[1] + (700 * math.sin(-angle * PI / 180.0))

        color = result(frame, x_i, y_i)

        # Draw ghost ball trajectory
        cv2.line(frame, (xc_center, yc_center), (x_sc, y_sc), color, LINE_THICKNESS)
        cv2.circle(frame, (x_sc, y_sc), GHOST_SIZE, color, cv2.FILLED)

        # Draw bounce trajectory
        cv2.line(frame, (xb_center, yb_center), (x_cb, y_cb), color, LINE_THICKNESS)

        cv2.line(frame, (intersection_point[0], intersection_point[1]), (int(x_i), int(y_i)), color, LINE_THICKNESS)
        cv2.circle(frame, (int(x_i), int(y_i)), GHOST_SIZE, color, cv2.FILLED)

        cv2.imshow('frame', frame)

        cv2.waitKey(WAIT_TIME)

    if success:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()
