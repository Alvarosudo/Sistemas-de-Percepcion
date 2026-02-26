import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) #Define los criterios de terminación para el proceso de refinamiento de las esquinas del tablero de ajedrez. El proceso se detendrá cuando se alcance un número máximo de iteraciones (30) o cuando la precisión de las esquinas refinadas sea menor que 0.001, lo que indica que se ha alcanzado una convergencia adecuada.

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32) #Crea un array de ceros con forma (54, 3) para almacenar las coordenadas 3D de los puntos del tablero de ajedrez. Cada fila representa un punto, y las tres columnas representan las coordenadas X, Y y Z en el espacio tridimensional.
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) #Crea una cuadrícula de puntos en el plano XY, con 9 columnas y 6 filas, y asigna las coordenadas a los primeros dos canales de objp. El tercer canal se mantiene en cero, lo que indica que todos los puntos están en el mismo plano Z=0.

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Used_Images/*.jpg')

for Practica1 in images:
    img = cv.imread(Practica1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        #cv.drawChessboardCorners(img, (9,6), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey()
cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret, mtx, dist, rvecs, tvecs)

new_img = cv.imread('WIN_20260216_15_40_46_Pro.jpg')

if new_img is None:
    print("Error: no se pudo cargar la imagen")
h,  w = new_img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(new_img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('Calib_Images/calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )