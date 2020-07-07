import cv2


def display_slices(img_slices):
    for i, img in enumerate(img_slices):
        cv2.imshow('Image Slice ' + str(i), img)
        cv2.waitKey(0)


def img_roi(img, upper_bound, lower_bound):
    cropped_img = img[upper_bound:-lower_bound, :]
    return cropped_img


def bboxes_included_in_crop(vertical, horizontal, interval, bboxes):
    for y, x, w, h in bboxes:
        cond_1 = (vertical <= y) and (y + w <= vertical + interval)
        cond_2 = (horizontal <= x) and (x + h <= horizontal + interval)
        if all([cond_1, cond_2]):
            return True

    return False


def img_slice_and_label(img, crop_size, bboxes):
    width = img.shape[1]
    height = img.shape[0]
    img_slices = []
    labels = []

    v_interval = crop_size - necessary_overlap_region(width, crop_size)
    h_interval = crop_size - necessary_overlap_region(height, crop_size)

    for vertical in range(0, width - crop_size, v_interval):
        for horizontal in range(0, height - crop_size, h_interval):
            crop = img[horizontal:horizontal + crop_size, vertical:vertical + crop_size]
            img_slices.append(crop)
            if bboxes_included_in_crop(vertical, horizontal, crop_size, bboxes):
                labels.append(1.0)
            else:
                labels.append(0.0)

    return img_slices, labels


def necessary_overlap_region(axis, interval):
    quotient, remainder = divmod(axis - interval, interval)
    overlap_region = int(remainder / (quotient))
    return overlap_region


if __name__ == '__main__':
    img_path = "D:/Razor Labs/Projects/AIS/data/RO2/RO2_OK_images/Cam1/img/PART1_PART1_Cam1_IO__23440-R02-C000_right_000154.png"
    img = cv2.imread(img_path, 0)
    # img = img_roi(img, *RO2_BOUNDS)
    img_slices, _ = img_slice_and_label(img, 256)
    print(len(img_slices))
    display_slices(img_slices)
