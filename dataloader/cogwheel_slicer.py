import cv2

RO2_BOUNDS = (20, 200)  # upper bound, lower bound
RO2_PERIOD = 200


def display_slices(img_slices):
    for i, img in enumerate(img_slices):
        cv2.imshow('Image Slice ' + str(i), img)
        cv2.waitKey(0)


def img_roi(img, upper_bound, lower_bound):
    cropped_img = img[upper_bound:-lower_bound, :]
    return cropped_img


def bboxes_included_in_crop(vertical, horizontal, interval, bboxes):
    for l, t, w, h in bboxes:
        cond_1 = (vertical <= l) and (l + w <= vertical + interval)
        cond_2 = (horizontal <= t) and (t + h <= horizontal + interval)
        if all([cond_1, cond_2]):
            return True

    return False


def img_slice_and_label(img, interval, bboxes):
    width = img.shape[1]
    height = img.shape[0]
    img_slices = []
    labels = []
    for vertical in range(0, width - interval, int(interval)):
        for horizontal in range(0, height - interval, int(interval)):
            img_slices.append(img[horizontal:horizontal + interval, vertical:vertical + interval])
            if bboxes_included_in_crop(vertical, horizontal, interval, bboxes):
                labels.append(1)
            else:
                labels.append(0)

    return img_slices, labels


def img_slicer(img, interval):
    width = img.shape[1]
    height = img.shape[0]
    img_slices = []
    for vertical in range(0, width - interval, interval):
        for horizontal in range(0, height - interval, interval):
            img_slices.append(img[horizontal:horizontal + interval, vertical:vertical + interval])

    return img_slices


if __name__ == '__main__':
    img_path = "D:/Razor Labs/Projects/AIS/data/RO2/RO2_OK_images/Cam1/img/PART1_PART1_Cam1_IO__23440-R02-C000_right_000154.png"
    img = cv2.imread(img_path, 0)
    img = img_roi(img, *RO2_BOUNDS)
    img_slices = img_slicer(img, 256)
    print(len(img_slices))
    # display_slices(img_slices)
