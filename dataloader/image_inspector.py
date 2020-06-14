import cv2

if __name__ == '__main__':
    img_path = "D:/Razor Labs/Projects/AIS/data/RO2/RO2_OK_images/Cam1/img/PART1_PART1_Cam1_IO__23440-R02-C000_right_000154.png"
    img = cv2.imread(img_path, 0)
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    print(img.shape)
    upper_bound, lower_bound, interval = 20, 200, 200
    sliced_img = img[upper_bound:-lower_bound, :interval]
    cv2.imshow('Image', sliced_img)
    cv2.waitKey(0)
