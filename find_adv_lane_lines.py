#!/usr/bin/env python
# coding=utf-8
# __author__='Alfred'

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import os.path
from collections import deque


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
    grad_dire_img = gradient_direction_filter(gray)

    # try using Color space transformation
    color_img = color_transform(img)

    # Composite binary images
    # binary_edges = ((sobelx_img == 1) | ((color_img == 1) & (grad_dire_img == 1))) * 1.0
    binary_edges = sobelx_img + ((color_img == 1) & (grad_dire_img == 1)) * 1.

    # plot gradient filter with color transformation
    # plot_img = np.dstack([np.zeros_like(sobelx_img), sobelx_img, color_img]) * 255
    # plt.figure(dpi=300)
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.title('Original Image')
    # plt.subplot(222)
    # plt.imshow(sobelx_img, cmap='gray')
    # plt.title('Sobel-x Image')
    # plt.subplot(223)
    # plt.imshow(color_img, cmap='gray')
    # plt.title('Color Image')
    # plt.subplot(224)
    # plt.imshow(binary_edges, cmap='gray')
    # plt.title('Binary edges of Image')
    #
    # plt.savefig("output_images/binary_edges.jpg")

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
        # dist_img = image_list[0]
        # un_dist = self.correct_image(dist_img)
        # plt.figure(dpi=300)
        # plt.subplot(121)
        # plt.imshow(dist_img)
        # plt.title('Original Image')
        #
        # plt.subplot(122)
        # plt.imshow(un_dist)
        # plt.title('undistorted Image')
        #
        # plt.savefig("output_images/un_dist.jpg")

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
        self._Minv = None

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
        self._Minv = cv2.getPerspectiveTransform(dst, src)

        # draw lines
        # line_color = [255, 0, 0]
        # thickness = 4
        # warped = self.warp_perspective(undist_img)
        #
        # cv2.line(undist_img, src_ltop, src_rtop, line_color, thickness)
        # cv2.line(undist_img, src_rtop, src_lbottom, line_color, thickness)
        # cv2.line(undist_img, src_lbottom, src_rbottom, line_color, thickness)
        # cv2.line(undist_img, src_rbottom, src_ltop, line_color, thickness)
        #
        # cv2.line(warped, dst_ltop, dst_rtop, line_color, thickness)
        # cv2.line(warped, dst_rtop, dst_lbottom, line_color, thickness)
        # cv2.line(warped, dst_lbottom, dst_rbottom, line_color, thickness)
        # cv2.line(warped, dst_rbottom, dst_ltop, line_color, thickness)
        #
        # plt.figure(dpi=300)
        # plt.subplot(121)
        # plt.imshow(undist_img)
        # plt.title('Original Image')
        # plt.subplot(122)
        # plt.imshow(warped)
        # plt.title('Perspective Image')
        # plt.savefig("output_images/perspective_trans.jpg")

    @property
    def M(self):
        return self._M

    @property
    def Minv(self):
        return self._Minv

    def warp_perspective(self, img):
        if len(img.shape) > 2:
            img_size = img.shape[:-1][::-1]
        elif len(img.shape) == 2:
            img_size = img.shape[::-1]
        else:
            img_size = None

        return cv2.warpPerspective(img, self._M, img_size, flags=cv2.INTER_LINEAR)

    def reverse_warp(self, img):
        if len(img.shape) > 2:
            img_size = img.shape[:-1][::-1]
        elif len(img.shape) == 2:
            img_size = img.shape[::-1]
        else:
            img_size = None

        return cv2.warpPerspective(img, self._Minv, img_size, flags=cv2.INTER_LINEAR)


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


def find_lane_line(img, perspective):
    """
    Finding lane lines in image
    :param img: original image
    :param perspective: Perspective class
    :return: a warped binary image
    """
    # copy image
    undist_img_backup = np.copy(img)
    # edges detection
    edges_img = edge_detection(undist_img_backup)
    # perspective transform
    perspective_img = perspective.warp_perspective(edges_img)

    return perspective_img


def sliding_window(img):
    out_img = np.dstack((img, img, img)) * 255

    x = img.shape[1]
    y = img.shape[0]

    histogram = np.sum(img[y // 4 * 3:, :], axis=0)
    leftx_base = np.argmax(histogram[: x // 2])
    rightx_base = np.argmax(histogram[x // 2:]) + x // 2

    # HYPER PARAMETERS
    # Choose the number of sliding windows
    nwin = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwin above and image shape
    win_h = np.int(y // nwin)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwin
    curr_leftx = leftx_base
    curr_rightx = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    def slid_win_by_pixel(x_low, x_high, y_low, y_high):
        inds = ((nonzerox >= x_low) & (nonzerox < x_high) & (nonzeroy >= y_low) & (nonzeroy < y_high)).nonzero()[0]
        return inds

    def draw_window(x_low, x_high, y_low, y_high):
        pass
        # cv2.rectangle(out_img, (x_low, y_low), (x_high, y_high), (0, 255, 0), 4)

    curr_n = 0
    while curr_n < nwin:
        win_y_low = y - (curr_n+1) * win_h
        win_y_high = y - curr_n * win_h
        win_xleft_low = curr_leftx - margin
        win_xleft_high = curr_leftx + margin
        win_xright_low = curr_rightx - margin
        win_xright_high = curr_rightx + margin

        left_inds = slid_win_by_pixel(win_xleft_low, win_xleft_high, win_y_low, win_y_high)
        right_inds = slid_win_by_pixel(win_xright_low, win_xright_high, win_y_low, win_y_high)

        # if the pixels detected is larger than minpix, choose this window
        if len(left_inds) > minpix:
            curr_leftx = int(np.mean(nonzerox[left_inds]))
            left_lane_inds.append(left_inds)
            draw_window(win_xleft_low, win_xleft_high, win_y_low, win_y_high)
        # slid the window left & right & not, compare which is best to choose
        else:
            slid_left_inds = slid_win_by_pixel(win_xleft_low - margin // 2, win_xleft_high - margin // 2, win_y_low,
                                               win_y_high)
            slid_right_inds = slid_win_by_pixel(win_xleft_low + margin // 2, win_xleft_high + margin // 2, win_y_low,
                                                win_y_high)

            chosen_idx = int(np.argmax([len(left_inds), len(slid_left_inds), len(slid_right_inds)]))
            chosen_inds = (left_inds, slid_left_inds, slid_right_inds)[chosen_idx]
            chosen_x_low, chosen_x_high = ((win_xleft_low, win_xleft_high),
                                           (win_xleft_low - margin // 2, win_xleft_high - margin // 2),
                                           (win_xleft_low + margin // 2, win_xleft_high + margin // 2))[chosen_idx]

            if len(chosen_inds) > 0:
                curr_leftx = int(np.mean(nonzerox[chosen_inds]))
                left_lane_inds.append(chosen_inds)
                draw_window(chosen_x_low, chosen_x_high, win_y_low, win_y_high)
            else:
                draw_window(win_xleft_low, win_xleft_high, win_y_low, win_y_high)

        # the same logic with above
        if len(right_inds) > minpix:
            curr_rightx = int(np.mean(nonzerox[right_inds]))
            right_lane_inds.append(right_inds)
            draw_window(win_xright_low, win_xright_high, win_y_low, win_y_high)
        else:
            slid_left_inds = slid_win_by_pixel(win_xright_low - margin // 2, win_xright_high - margin // 2, win_y_low,
                                               win_y_high)
            slid_right_inds = slid_win_by_pixel(win_xright_low + margin // 2, win_xright_high + margin // 2, win_y_low,
                                                win_y_high)

            chosen_idx = int(np.argmax([len(right_inds), len(slid_left_inds), len(slid_right_inds)]))
            chosen_inds = (right_inds, slid_left_inds, slid_right_inds)[chosen_idx]
            chosen_x_low, chosen_x_high = ((win_xright_low, win_xright_high),
                                           (win_xright_low - margin // 2, win_xright_high - margin // 2),
                                           (win_xright_low + margin // 2, win_xright_high + margin // 2))[chosen_idx]

            if len(chosen_inds) > 0:
                curr_rightx = int(np.mean(nonzerox[chosen_inds]))
                right_lane_inds.append(chosen_inds)
                draw_window(chosen_x_low, chosen_x_high, win_y_low, win_y_high)
            else:
                draw_window(win_xright_low, win_xright_high, win_y_low, win_y_high)

        curr_n += 1

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # plot sliding windows
    # plt.figure(dpi=300)
    # ploty = np.linspace(0, y-1, y)
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    #
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    #
    # plt.savefig("output_images/sliding_window.jpg")

    return left_fit, right_fit, leftx, lefty, rightx, righty


def search_from_poly(img):
    # HYPER PARAMETER
    margin = 100

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_p = np.poly1d(LEFT_LINE.current_fit)
    right_p = np.poly1d(RIGHT_LINE.current_fit)

    leftx_low = left_p(nonzeroy) - margin
    leftx_high = left_p(nonzeroy) + margin
    rightx_low = right_p(nonzeroy) - margin
    rightx_high = right_p(nonzeroy) + margin

    left_lane_inds = ((nonzerox >= leftx_low) & (nonzerox < leftx_high)).nonzero()[0]
    right_lane_inds = ((nonzerox >= rightx_low) & (nonzerox < rightx_high)).nonzero()[0]

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, leftx, lefty, rightx, righty


REAL_LANE_LINE_HEIGHT = 30
REAL_LANE_LINE_WIDTH = 3.7


def calc_curvature_and_position(left_fit, right_fit, img):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = REAL_LANE_LINE_HEIGHT / img.shape[0]  # meters per pixel in y dimension
    xm_per_pix = REAL_LANE_LINE_WIDTH / img.shape[1]  # meters per pixel in x dimension

    # rewrite the polynomial with the conversions
    ori_la, ori_lb = left_fit[:-1]
    ori_ra, ori_rb = right_fit[:-1]
    # determine the direction
    left_dire = 1 if ori_la > 0 else -1
    right_dire = 1 if ori_ra > 0 else -1

    left_a = xm_per_pix / (ym_per_pix ** 2) * ori_la
    left_b = xm_per_pix / ym_per_pix * ori_lb
    right_a = xm_per_pix / (ym_per_pix ** 2) * ori_ra
    right_b = xm_per_pix / ym_per_pix * ori_rb

    # using the bottom pixel of the image to calculate the curvature and position.
    y_evel = (img.shape[0] - 1)
    y_real = y_evel * ym_per_pix

    left_curvature = left_dire * (1 + (2 * left_a * y_real + left_b) ** 2) ** (3 / 2) / np.absolute(2 * left_a)
    right_curvature = right_dire * (1 + (2 * right_a * y_real + right_b) ** 2) ** (3 / 2) / np.absolute(2 * right_a)

    # calculate the position
    left_pos = np.poly1d(left_fit)(y_evel)
    right_pos = np.poly1d(right_fit)(y_evel)
    mid_pos = (left_pos + right_pos) / 2

    position = (img.shape[1]/2 - mid_pos) * xm_per_pix

    return left_curvature, right_curvature, position


# Define a class to receive the characteristics of each line detection
class Line(object):
    def __init__(self, last_frames):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # count how many frames losing the lines in a row
        self.lost_count = 0
        # pixels of the last n fits of the line
        self.recent_fitx = deque(maxlen=last_frames)
        self.recent_fity = deque(maxlen=last_frames)


# tracking left & right lane line
# HYPER PARAMETER
LAST_FRAMES = 5
LEFT_LINE = Line(LAST_FRAMES)
RIGHT_LINE = Line(LAST_FRAMES)

# HYPER PARAMETERS
MAX_CURVATURE = 9000
MAX_POSITION = 0.3
MAX_DERIVATIVE_ERROR = 2e-1


def sanity_check(img, left_fit, right_fit):
    x = img.shape[1]
    y = img.shape[0]

    # check the direction of left_curvature and right_curvature
    lcur, rcur, pos = calc_curvature_and_position(left_fit, right_fit, img)
    if lcur * rcur >= 0:
        if np.abs(lcur) > MAX_CURVATURE and np.abs(rcur) > MAX_CURVATURE:
            # treated as two straight lines
            pass
        else:
            if np.abs(lcur - rcur) > np.abs(lcur + rcur) * .5:  # 211~859
                return False
    else:
        if np.abs(lcur) < MAX_CURVATURE or np.abs(rcur) < MAX_CURVATURE:
            return False
    # check the vehicle position
    if np.abs(pos) > MAX_POSITION:
        return False

    # Checking lines are separated by approximately the right distance horizontally
    left_base_x = np.poly1d(left_fit)(y)
    right_base_x = np.poly1d(right_fit)(y)
    if not (0.4 * x <= (right_base_x - left_base_x) <= 0.7 * x):
        return False

    # Checking that they are roughly parallel, using top/middle/bottom y-axis value's derivative
    y_checkpoint = [0, y/2, y]
    left_a, left_b, _ = left_fit
    right_a, right_b, _ = right_fit

    for yc in y_checkpoint:
        left_dx = 2 * left_a * yc + left_b
        right_dx = 2 * right_a * yc + right_b

        if np.abs(left_dx - right_dx) > MAX_DERIVATIVE_ERROR:  # np.abs(left_dx + right_dx) * .5:
            return False

    return True


def line_smoothing(leftx, lefty, rightx, righty):
    LEFT_LINE.recent_fitx.append(leftx)
    LEFT_LINE.recent_fity.append(lefty)
    RIGHT_LINE.recent_fitx.append(rightx)
    RIGHT_LINE.recent_fity.append(righty)

    all_leftx = np.concatenate(LEFT_LINE.recent_fitx)
    all_lefty = np.concatenate(LEFT_LINE.recent_fity)
    all_rightx = np.concatenate(RIGHT_LINE.recent_fitx)
    all_righty = np.concatenate(RIGHT_LINE.recent_fity)

    left_fit = np.polyfit(all_lefty, all_leftx, 2)
    right_fit = np.polyfit(all_righty, all_rightx, 2)

    LEFT_LINE.current_fit = left_fit
    RIGHT_LINE.current_fit = right_fit

    return left_fit, right_fit


def predict_lane_line(img):
    if LEFT_LINE.detected and RIGHT_LINE.detected:
        lfit, rfit, leftx, lefty, rightx, righty = search_from_poly(img)
    else:
        # fit the lane lines using sliding window
        lfit, rfit, leftx, lefty, rightx, righty = sliding_window(img)

    # Sanity Check
    if sanity_check(img, lfit, rfit):
        LEFT_LINE.detected = True
        RIGHT_LINE.detected = True
        LEFT_LINE.lost_count = 0
        RIGHT_LINE.lost_count = 0

        # add lane line smoothing, recalculate left_fit & right_fit
        left_fit, right_fit = line_smoothing(leftx, lefty, rightx, righty)
    else:
        left_fit, right_fit = LEFT_LINE.current_fit, RIGHT_LINE.current_fit

        if LEFT_LINE.lost_count >= LAST_FRAMES or RIGHT_LINE.lost_count >= LAST_FRAMES:
            LEFT_LINE.detected = False
            RIGHT_LINE.detected = False
            LEFT_LINE.lost_count += 1
            RIGHT_LINE.lost_count += 1
        else:
            LEFT_LINE.lost_count += 1
            RIGHT_LINE.lost_count += 1

    # ---------------- returning results ---------------- #

    # calculate the curvature and position
    left_curvature, right_curvature, position = calc_curvature_and_position(left_fit, right_fit, img)
    # combine left_curvature & right_curvature
    if left_curvature * right_curvature < 0:
        curvature = 0
    else:
        curvature = (left_curvature + right_curvature) / 2
    # if the curvature is too large, the lane lines are nearly straight
    curvature = 0 if np.absolute(curvature) > MAX_CURVATURE else curvature

    return left_fit, right_fit, curvature, position


def drawing_lane_line_area(undist, warped_img, perspective, left_fit, right_fit, curvature, position):
    y, x = undist.shape[:-1]

    ploty = np.linspace(0, y-1, y)
    left_plotx = np.poly1d(left_fit)(ploty)
    right_plotx = np.poly1d(right_fit)(ploty)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_plotx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix
    new_warp = perspective.reverse_warp(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)

    # Write prediction results on image
    out_format = 'Cur: %.3f(m) %s, Pos: %.3f(m) %s'
    if curvature < 0:
        cur_dire = 'left'
    elif curvature > 0:
        cur_dire = 'right'
    else:
        cur_dire = 'straight'

    if position > 0:
        pos_dire = 'right'
    elif position < 0:
        pos_dire = 'left'
    else:
        pos_dire = 'straight'

    out_str = out_format % (np.abs(curvature), cur_dire, np.abs(position), pos_dire)
    cv2.putText(result, out_str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    return result


# get camera & perspective instance
camera, perspective = pre_process()


def my_pipeline_with_img(img):
    # correct image
    undist_img = camera.correct_image(img)
    # finding lane lines
    lane_line_img = find_lane_line(undist_img, perspective)
    # predicting lane line curvature and the vehicle position
    left_fit, right_fit, curvature, position = predict_lane_line(lane_line_img)
    # plot result back down onto the road
    result_img = drawing_lane_line_area(undist_img, lane_line_img, perspective, left_fit, right_fit, curvature, position)

    return result_img


def my_pipeline_with_video(video_fp):
    fn = os.path.basename(video_fp).split('.')[0]
    white_output = 'output_videos/%s_result.mp4' % fn

    # optimize lane lines detection in video
    # clip1 = VideoFileClip(video_fp).subclip(0, 10)
    clip1 = VideoFileClip(video_fp)
    res_clip = clip1.fl_image(my_pipeline_with_img)
    res_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    # testing images
    # test_images = list(map(mpimg.imread, glob.glob('test_images/*.jpg')))
    # img_idx = -2
    # # img_idx = np.random.choice(list(range(len(test_images))))
    # test_img = test_images[img_idx]
    # res_img = my_pipeline_with_img(test_img)
    #
    # plt.imshow(res_img)
    # plt.show()

    # testing videos
    fp = 'project_video.mp4'
    my_pipeline_with_video(fp)
