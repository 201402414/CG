import cv2
import numpy as np

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

# filter와 image를 입력받아 filtering수행
def my_filtering(src, filter):
    (h, w) = src.shape
    (m_h, m_w) = filter.shape
    pad_img =my_padding(src, filter)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * filter)
    return dst

# sobel filter 생성
def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    print('sobel x')
    print(sobel_x)
    print('sobel y')
    print(sobel_y)
    return sobel_x, sobel_y

# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 되지만, 과제 제출 시 DoG 사용하면 감점

    #가우시안 필터
    filter1D = cv2.getGaussianKernel(fsize, sigma)
    filter = np.dot(filter1D, filter1D.T)
    #print(filter)
    dst = my_filtering(src, filter)

    # sobel filter sobel filter의 경우 3x3 sobel filter만 실습때 사용함
    sobel_x, sobel_y = get_my_sobel()
    Ix = my_filtering(dst, sobel_x)
    Iy = my_filtering(dst, sobel_y)

    return Ix, Iy


# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                   #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    temp = Ix**2 + Iy**2
    magnitude = np.sqrt(temp)

    return magnitude


# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    #######################################
    # TODO                               #
    # calcAngle 완성                      #
    # angle     : ix와 iy의 angle         #
    #######################################
    radian_angle = np.arctan(Iy/Ix)

    angle = np.rad2deg(radian_angle)


    return angle


# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # larger_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    (h, w) = magnitude.shape
    cv2.imshow('before non maximum supression', magnitude/255)
    # angle의 범위 : -90 ~ 90
    larger_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]

            # gradient의 degree는 edge와 수직방향이다.
            if 0 <= degree and degree < 45:
                rate = np.tan(np.deg2rad(degree))
                # ********************
                left_magnitude = (rate) * magnitude[row - 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
                right_magnitude = (rate) * magnitude[row + 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            elif -45 > degree and degree >= -90:
                rate = -(1/np.tan(np.deg2rad(degree)))
                up_magnitude = (rate) * magnitude[row - 1, col + 1] + (1 - rate) * magnitude[row - 1, col]
                down_magnitude = (rate) * magnitude[row + 1, col - 1] + (1 - rate) * magnitude[row + 1, col]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            elif -45 <= degree and degree < 0:
                rate = -np.tan(np.deg2rad(degree))
                left_magnitude = (rate) * magnitude[row + 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
                right_magnitude = (rate) * magnitude[row - 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            elif 90 >= degree and degree >= 45:
                rate = 1/np.tan(np.deg2rad(degree))
                up_magnitude = (rate) * magnitude[row - 1, col - 1] + (1 - rate) * magnitude[row - 1, col]
                down_magnitude = (rate) * magnitude[row + 1, col + 1] + (1 - rate) * magnitude[row + 1, col]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    larger_magnitude[row, col] = magnitude[row, col]

            else:
                # angle을 np.arctna(Iy/Ix) 로 구했는데 Ix값이 0일 경우 해당 angle은 nan값이 저장됨
                print(row, col, 'error!  degree :', degree)

    larger_magnitude = (larger_magnitude / np.max(larger_magnitude) * 255).astype(np.uint8)
    cv2.imshow('after non maximum supression', larger_magnitude)
    return larger_magnitude


# double_thresholding 수행 high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고 low threshold값은 (high threshold * 0.4)로 구한다
def double_thresholding(src, test_mode=True):
    (h, w) = src.shape
    high_threshold_value, _ = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    print('highthreshold')
    print(high_threshold_value)
    if test_mode == True:
        print('test mode!! - double threshold function')
        high_threshold_value = 200
    low_threshold_value = high_threshold_value * 0.4
    cv2.THRESH_BINARY
    dst = src.copy()

    for row in range(h):
        for col in range(w):
            if dst[row, col] >= high_threshold_value:
                dst[row, col] = 255
            elif dst[row, col] < low_threshold_value:
                dst[row, col] = 0
            else:
                ##################################################
                # TODO                                           #
                # high 보다는 작고 low보다는 큰 경우                 #
                ##################################################
                dst[row, col] = 127

    high_edge = np.sum(_ == 255)

    while (1):
        mask(dst)
        if (high_edge == np.sum(dst == 255)):
            break
        high_edge = np.sum(dst == 255)

    for row in range(h):
        for col in range(w):
            if (dst[row, col] == 127):
                dst[row, col] = 0

    return dst

def mask(src) :
    (h, w) = src.shape
    for i in range(h) :
        for j in range(w) :
            if(src[i, j] == 127) :
                t_max = max(src[i-1, j-1], src[i-1, j], src[i-1, j+1], src[i, j-1],
                            src[i, j+1], src[i+1, j-1], src[i+1, j], src[i+1, j+1])
                if(t_max == 255) :
                    src[i, j] = 255

def my_canny_edge_detection(src, fsize=3, sigma=1, test_mode=False):
    if test_mode == False:
        # low-pass filter를 이용하여 blur효과
        # high-pass filter를 이용하여 edge 검출
        # gaussian filter -> sobel filter 를 이용해서 2번 filtering
        Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)
        cv2.imshow("Ix", Ix/255)
        cv2.imshow("Iy", Iy/255)

        # magnitude와 angle을 구함
        magnitude = calcMagnitude(Ix, Iy)
        angle = calcAngle(Ix, Iy)

        # non-maximum suppression 수행
        larger_magnitude = non_maximum_supression(magnitude, angle)

        # double thresholding 수행
        dst = double_thresholding(larger_magnitude, test_mode=test_mode)

    elif test_mode == True:
        dst = double_thresholding(src, test_mode=test_mode)
        cv2.imwrite('double_threshold_test_result.png', dst)

    return dst


if __name__ == '__main__':
    # double threshold test
    src = cv2.imread('double_threshold_testImg.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src, test_mode=True)

    # src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # dst = my_canny_edge_detection(src)
    #
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
