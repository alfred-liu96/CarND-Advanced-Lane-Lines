#!/usr/bin/env python
# coding=utf-8
# __author__='Alfred'

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def canny_binary_edges(img):
    canny_low = 50
    canny_high = 120

    edges = cv2.Canny(img, canny_low, canny_high)
    binary_edges = np.zeros_like(edges)
    binary_edges[edges > 0] = 1

    return binary_edges


def gradient_magnitude_filter(img):
    sobel_low = 20
    sobel_high = float('inf')  # 100
    sobel_size = 3

    # calculate gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, img.shape, sobel_size)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.float32(255 * abs_sobelx / np.max(abs_sobelx))
    sobelx_edges = np.zeros_like(scaled_sobelx)
    sobelx_edges[(scaled_sobelx >= sobel_low) & (scaled_sobelx <= sobel_high)] = 1

    return sobelx_edges


def gradient_direction_filter(img):
    direction_low = .7  # 0.7, 40/180 * np.pi
    direction_high = 1.3  # 1.3, 90/180 * np.pi
    sobel_size = 15

    # calculate gradient direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, img.shape, sobel_size)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, img.shape, sobel_size)
    abs_sobely = np.absolute(sobely)
    direction_gradient = np.arctan2(abs_sobely, abs_sobelx)
    direction_edges = np.zeros_like(direction_gradient)
    direction_edges[(direction_gradient >= direction_low) & (direction_gradient <= direction_high)] = 1

    return direction_edges


def color_transform(img):
    color_low = 170
    color_high = 255

    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_img = hls_img[:, :, 2]

    bin_color = np.zeros_like(s_img)
    bin_color[(s_img >= color_low) & (s_img <= color_high)] = 1
    return bin_color


def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    """
    try using the Canny transform, but it doesn't work well,
    because there is too much noise in the y-axis which influences the lane line detection
    """
    # canny_img = canny_binary_edges(gray)

    # try using gradient filter
    sobelx_img = gradient_magnitude_filter(gray)
    # TODO tune the gradient-direction's params to get a better result
    grad_dire_img = gradient_direction_filter(gray)

    # try using Color space transformation
    color_img = color_transform(img)

    # Composite binary images
    # binary_edges = ((sobelx_img == 1) | ((color_img == 1) & (grad_dire_img == 1))) * 1.0
    binary_edges = sobelx_img + ((color_img == 1) & (grad_dire_img == 1)) * 1.

    # plot gradient filter with color transformation
    # plot_img = np.dstack([np.zeros_like(sobelx_img), sobelx_img, color_img]) * 255
    plt.figure(dpi=300)
    plt.subplot(221)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(222)
    plt.imshow(sobelx_img, cmap='gray')
    plt.title('Sobel-x Image')
    plt.subplot(223)
    plt.imshow(color_img, cmap='gray')
    plt.title('Color Image')
    plt.subplot(224)
    plt.imshow(binary_edges, cmap='gray')
    plt.title('Binary edges of Image')

    plt.savefig("output_images/binary_edges.jpg")

    return binary_edges


class Camera(object):
    def __init__(self):
        self._mtx = None
        self._dist = None

    def camera_calibration(self, image_list):
        nx = 9
        ny = 6
        obj_p = np.zeros((nx * ny, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        img_size = image_list[0].shape[:-1]
        obj_points = []
        img_points = []
        for img in image_list:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
            if ret is True:
                obj_points.append(obj_p)
                img_points.append(corners)

        ret, self._mtx, self._dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size[::-1], None, None)

        # use plt to add two images in one picture, and save it to disk
        dist_img = image_list[0]
        un_dist = self.correct_image(dist_img)
        plt.figure(dpi=300)
        plt.subplot(121)
        plt.imshow(dist_img)
        plt.title('Original Image')

        plt.subplot(122)
        plt.imshow(un_dist)
        plt.title('undistorted Image')

        plt.savefig("output_images/un_dist.jpg")

    @property
    def mtx(self):
        return self._mtx

    @property
    def dist(self):
        return self._dist

    def correct_image(self, img):
        return cv2.undistort(img, self._mtx, self._dist)


class Perspective(object):
    def __init__(self):
        self._M = None

    def gen_perspective_matrix(self, camera):
        # using an image where the lane lines are straight to tune the positions
        img = mpimg.imread("test_images/straight_lines1.jpg")
        undist_img = camera.correct_image(img)

        y, x = undist_img.shape[:-1]
        src_ltop = (x // 2 - 52, y // 2 + 95)
        src_rtop = (x // 2 + 52, y // 2 + 95)
        src_lbottom = (x - 180, y)
        src_rbottom = (180, y)
        src = np.float32(
            [
                [src_ltop],
                [src_rtop],
                [src_lbottom],
                [src_rbottom]
            ]
        )

        offset_x = 300
        dst_ltop = (offset_x, 0)
        dst_rtop = (x - offset_x, 0)
        dst_lbottom = (x - offset_x, y)
        dst_rbottom = (offset_x, y)
        dst = np.float32(
            [
                [dst_ltop],
                [dst_rtop],
                [dst_lbottom],
                [dst_rbottom]
            ]
        )

        self._M = cv2.getPerspectiveTransform(src, dst)

        # draw lines
        line_color = [255, 0, 0]
        thickness = 4
        warped = self.warp_perspective(undist_img)

        cv2.line(undist_img, src_ltop, src_rtop, line_color, thickness)
        cv2.line(undist_img, src_rtop, src_lbottom, line_color, thickness)
        cv2.line(undist_img, src_lbottom, src_rbottom, line_color, thickness)
        cv2.line(undist_img, src_rbottom, src_ltop, line_color, thickness)

        cv2.line(warped, dst_ltop, dst_rtop, line_color, thickness)
        cv2.line(warped, dst_rtop, dst_lbottom, line_color, thickness)
        cv2.line(warped, dst_lbottom, dst_rbottom, line_color, thickness)
        cv2.line(warped, dst_rbottom, dst_ltop, line_color, thickness)

        plt.figure(dpi=300)
        plt.subplot(121)
        plt.imshow(undist_img)
        plt.title('Original Image')
        plt.subplot(122)
        plt.imshow(warped)
        plt.title('Perspective Image')
        plt.savefig("output_images/perspective_trans.jpg")

    @property
    def M(self):
        return self._M

    def warp_perspective(self, img):
        if len(img.shape) > 2:
            img_size = img.shape[:-1][::-1]
        elif len(img.shape) == 2:
            img_size = img.shape[::-1]
        else:
            img_size = None

        return cv2.warpPerspective(img, self._M, img_size, flags=cv2.INTER_LINEAR)


def pre_process():
    """
    Prepare for lane line detection, including Camera-Calibration, Perspective-Transformation
    :return: Camera class & Perspective class
    """
    camera = Camera()
    camera_cal_images = list(map(mpimg.imread, glob.glob('camera_cal/calibration*.jpg')))
    camera.camera_calibration(camera_cal_images)

    perspective = Perspective()
    perspective.gen_perspective_matrix(camera)

    return camera, perspective


def find_lane_line(img, camera, perspective):
    """
    Finding lane lines in image
    :param img: original image
    :param camera: Camera class
    :param perspective: Perspective class
    :return: a warped binary image
    """
    # correct image
    undist_img = camera.correct_image(img)
    # copy image
    undist_img_backup = np.copy(undist_img)
    # edges detection
    edges_img = edge_detection(undist_img_backup)
    # perspective transform
    perspective_img = perspective.warp_perspective(edges_img)

    return perspective_img


class Line(object):
    def __init__(self):
        pass


def sliding_windows(img):
    histogram = np.sum(img[img.shape[0] // 4:, :], axis=0)


def predict_lane_line(img):
    out_img = np.dstack((img, img, img)) * 255

    return 0, 0, out_img


def my_pipeline_with_img(img):
    # get camera & perspective instance
    camera, perspective = pre_process()
    # finding lane lines
    lane_line_img = find_lane_line(img, camera, perspective)
    # predicting lane line curvature and the vehicle position
    curvature, position, out_img = predict_lane_line(lane_line_img)

    return curvature, position, lane_line_img


def my_pipeline_with_video():
    # TODO deal with videos
    pass


if __name__ == '__main__':
    # testing images
    test_images = list(map(mpimg.imread, glob.glob('test_images/*.jpg')))
    img_idx = -2
    # img_idx = np.random.choice(list(range(len(test_images))))
    test_img = test_images[img_idx]
    cur, pos, result_img = my_pipeline_with_img(test_img)

    plt.figure(dpi=300)
    plt.subplot(121)
    plt.imshow(test_img)
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(result_img, cmap='gray')
    plt.title('Processed Image')
    plt.show()

    # TODO testing videos

    # TODO comment all plot funcs
