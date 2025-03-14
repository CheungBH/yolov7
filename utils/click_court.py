import cv2

def click_points(img):
    mask_points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mask_points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('image', img)
            if len(mask_points) > 4:
                mask_points.pop(0)
            print(mask_points)

    height, width, channel = img.shape
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", width, height)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_event)

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    return mask_points
