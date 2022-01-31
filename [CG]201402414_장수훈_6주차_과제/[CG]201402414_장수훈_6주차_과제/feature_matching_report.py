import cv2
import numpy as np
import random

def feature_matching(img1, img2, RANSAC=False, threshold = 300, keypoint_num = None, iter_num = 500, threshold_distance=10):

    sift = cv2.xfeatures2d.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    distance = []
    for idx_1, des_1 in enumerate(des1):
        dist = []
        for idx_2, des_2 in enumerate(des2):
            dist.append(L2_distance(des_1, des_2))

        distance.append(dist)

    distance = np.array(distance)

    min_dist_idx = np.argmin(distance, axis=1)
    min_dist_value = np.min(distance, axis=1)

    points = []
    for idx, point in enumerate(kp1):
        if min_dist_value[idx] >= threshold:
            continue

        x1, y1 = point.pt
        x2, y2 = kp2[min_dist_idx[idx]].pt

        x1 = int(np.round(x1))
        y1 = int(np.round(y1))

        x2 = int(np.round(x2))
        y2 = int(np.round(y2))
        points.append([(x1, y1), (x2, y2)])


    # no RANSAC
    if not RANSAC:

        A = []
        B = []
        for idx, point in enumerate(points):
            '''
            #ToDo
            #A, B 완성
            # A.append(???) 이런식으로 할 수 있음
            # 결과만 잘 나오면 다른방법으로 해도 상관없음
            '''
            A.append([points[idx][0][0], points[idx][0][1], 1, 0, 0, 0])
            A.append([0, 0, 0, points[idx][0][0], points[idx][0][1], 1])
            B.append(points[idx][1][0])
            B.append(points[idx][1][1])

        A = np.array(A)
        B = np.array(B)

        '''
        #ToDo
        #X 완성
        #np.linalg.inv(V) : V의 역행렬 구하는것
        #np.dot(V1, V2) : V1과 V2의 행렬곱
        # V1.T : V1의 transpose
        '''

        X = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),B)

        '''
        # ToDo
        # 위에서 구한 X를 이용하여 M 완성
        '''
        M = [[X[0], X[1], X[2]],
             [X[3], X[4], X[5]],
             [0   , 0   ,    1]]

        M = np.array(M)
        M_ = np.linalg.inv(M)

        # print(M)
        # print(M_)

        '''
        # ToDo
        # backward 방식으로 dst완성
        '''
        #Backward 방식

        h, w = img1.shape[:2]
        dst = np.zeros((h, w,3))
        count = dst.copy()

        for row in range(h):
            for col in range(w):
                vec = np.dot(M_, np.array([[col,row,1]]).T)
                c = vec[0,0]
                r = vec[1,0]
                c_left = int(c)
                c_right = min(int(c+1),w-1)
                r_top = int(r)
                r_bottom = min(int(r+1),h-1)
                s = c - c_left
                t = r - r_top

                intensity = (1-s) * (1-t) * img1[r_top,c_left] + s * (1-t) * img1[r_top,c_right] + (1-s) * t * img1[r_bottom,c_left] + s * t * img1[r_bottom,c_right]
                dst[row, col] = intensity
        dst = dst.astype(np.uint8)


        # for row in range(h):
        #     for col in range(w):
        #         vec = np.dot(M, np.array([[col, row, 1]]).T)
        #         x = vec[0, 0]
        #         y = vec[1, 0]
        #
        #         x1 = int(np.floor(x))
        #         x2 = int(np.ceil(x))
        #         y1 = int(np.floor(y))
        #         y2 = int(np.ceil(y))
        #
        #         points_list = [(y1,x1),(y1,x2), (y2,x1), (y2,x2)]
        #         points = set(points_list)
        #
        #         for (row_, col_) in points:
        #             dst[min(row_, h ), min(col_, w )] += img1[row, col]
        #             count[min(row_, h ), min(col_, w )] += 1
        #
        # dst = (dst / count).astype(np.uint8)

    #use RANSAAC
    # else:
    #     points_shuffle = points.copy()
    #
    #     inliers = []
    #     M_list = []
    #     for i in range(iter_num):
    #         random.shuffle(points_shuffle)
    #         three_points = points_shuffle[:3]
    #
    #         A = []
    #         B = []
            #3개의 point만 가지고 M 구하기
            # for idx, point in enumerate(three_points):
            #     '''
            #     #ToDo
            #     #A, B 완성
            #     # A.append(???) 이런식으로 할 수 있음
            #     # 결과만 잘 나오면 다른방법으로 해도 상관없음
            #     '''
            #     A.append([points[idx][0][0], points[idx][0][1], 1, 0, 0, 0])
            #     A.append([0, 0, 0, points[idx][0][0], points[idx][0][1], 1])
            #     B.append(points[idx][1][0])
            #     B.append(points[idx][1][1])
            #
            # A = np.array(A)
            # B = np.array(B)
    #         try:
    #             '''
    #             #ToDo
    #             #X 완성
    #             #np.linalg.inv(V) : V의 역행렬 구하는것
    #             #np.dot(V1, V2) : V1과 V2의 행렬곱
    #             # V1.T : V1의 transpose 단, type이 np.array일때만 가능. type이 list일때는 안됨
    #             '''
    #             X = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),B)
    #
    #         except:
    #             print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
    #             continue
    #
    #         '''
    #         # ToDo
    #         # 위에서 구한 X를 이용하여 M 완성
    #         '''
    #         M = [[X[0], X[1], X[2]],
    #              [X[3], X[4], X[5]],
    #              [0   , 0   , 1   ]]
    #
    #         M_list.append(M)
    #
    #         count_inliers = 0
    #         for idx, point in enumerate(points):
    #             '''
    #             # ToDo
    #             # 위에서 구한 M으로(3개의 point로 만든 M) 모든 point들에 대하여 예상 point 구하기
    #             # 구해진 예상 point와 실제 point간의 L2 distance 를 구해서 threshold_distance보다 작은 값이 있는 경우 inlier로 판단
    #             '''
    #             ???(M으로 구한 point)
    #             ???(실제 point)
    #             if L2_distance(???(M으로 구한 point), ???(실제 point)) < threshold_distance:
    #                 count_inliers += 1
    #
    #         inliers.append(count_inliers)
    #
    #     inliers = np.array(inliers)
    #     max_inliers_idx = np.argmax(inliers)
    #
    #     best_M = np.array(M_list[max_inliers_idx])
    #
    #     M = best_M
    #     M_ = np.linalg.inv(M)
    #
    #
    #     '''
    #     # ToDo
    #     # backward 방식으로 dst완성
    #     '''
    #     #Backward 방식
    #     dst = ???

    return dst


def L2_distance(vector1, vector2):
    '''
    #vector1과 vector2의 거리 구하기 (L2 distance)
    #distance 는 스칼라
    #np.sqrt(), np.sum() 를 잘 활용하여 구하기
    #L2 distance를 구하는 내장함수로 거리를 구한 경우 감점
    '''
    # distance = np.sqrt(np.sum(np.dot(vector1-vector2,vector1-vector2)))
    distance = np.sqrt(np.sum((vector1 - vector2)**2))
    # distance2 = np.linalg.norm(vector1-vector2)
    # if distance != distance2:
    #     print('ffff')

    # print(distance)
    return distance


def main():
    img = cv2.imread('./building.jpg')
    img_ref = cv2.imread('./building_temp.jpg')

    threshold = 300
    iter_num = 500
    #속도가 너무 느리면 100과 같이 숫자로 입력
    # keypoint_num = None
    keypoint_num = 50
    threshold_distance = 10

    dst_no_ransac = feature_matching(img, img_ref, threshold=threshold, keypoint_num=keypoint_num, iter_num=iter_num, threshold_distance=threshold_distance)
    # dst_use_ransac = feature_matching(img, img_ref, RANSAC=True, threshold=threshold, keypoint_num=keypoint_num, iter_num=iter_num, threshold_distance=threshold_distance)

    cv2.imshow('No RANSAC' + '201402414', dst_no_ransac)
    # cv2.imshow('Use RANSAC' + 201402414, dst_use_ransac)

    cv2.imshow('original image' + '201402414', img)
    cv2.imshow('reference image' + '201402414', img_ref)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ =='__main__':
    main()


