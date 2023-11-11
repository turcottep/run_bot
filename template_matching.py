import cv2
import numpy as np

screenshot = cv2.imread('last.png')
key_img = cv2.imread('right_light.png')

w = key_img.shape[1]
h = key_img.shape[0]


# cv2.imshow('Farm', farm_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.imshow('Needle', wheat_img)
# cv2.waitKey()
# cv2.destroyAllWindows()


result = cv2.matchTemplate(screenshot, key_img, cv2.TM_SQDIFF_NORMED)


result_clean = result * 255
cv2.imwrite('result.png', result_clean)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print("min_val: ", min_val, "max_val: ", max_val, "min_loc: ", min_loc, "max_loc: ", max_loc)

# get second best match with at least 100 pixel distance
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result, mask=np.where(result > min_val + 0.01, 0, 1))


# image shows as black, but it is not

# cv2.imshow('result', result)
# cv2.waitKey()
# cv2.destroyAllWindows()

# threshold = 0.025
# yloc, xloc = np.where(result < threshold)


top_left = min_loc

bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(screenshot, top_left, bottom_right, 255, 10)

top_left2 = min_loc2

bottom_right2 = (top_left2[0] + w, top_left2[1] + h)
cv2.rectangle(screenshot, top_left2, bottom_right2, 255, 10)


# rectangles = []
# for (x, y) in zip(xloc, yloc):
#     rectangles.append([int(x), int(y), int(w), int(h)])
#     rectangles.append([int(x), int(y), int(w), int(h)])

# rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

# for (x, y, w, h) in rectangles:
#     cv2.rectangle(screenshot, (x, y), (x + w, y + h), (255, 0, 0), 1)

cv2.imwrite('screenshot_with_result.png', screenshot)

# cv2.imshow('screenshot', screenshot)
# cv2.waitKey()
# cv2.destroyAllWindows()
