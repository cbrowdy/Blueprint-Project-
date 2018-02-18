import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Face_Match")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Face_Match", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "user.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("user written!".format(img_name))
        img_counter += 1
        break

cam.release()

cv2.destroyAllWindows()