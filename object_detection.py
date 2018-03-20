import cv2
import numpy as np
import dlt
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA

def transform(H,points):
    transInliers = np.ones((len(points),3))
    j=0
    for point in points:
        transInliers [j] =  H @ point
        transInliers [j][0] /= transInliers[j][2]
        transInliers [j][1] /= transInliers[j][2]
        transInliers [j][2] /= transInliers[j][2]
        j+=1
    return transInliers

img1 = cv2.imread("img1.jpg",0)
img2 = cv2.imread("img2.jpg",0)

sift = cv2.xfeatures2d.SIFT_create()


kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)


bf = cv2.BFMatcher(normType=cv2.NORM_L2,crossCheck=True)
matches = bf.match(des1,des2)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

"""
orb = cv2.ORB_create(WTA_K=4,nfeatures=1800)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
matches = bf.match(des1, des2)
"""

good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])



firstOrgs = []
firstCorrs = []


for m in good:
    img1_idx = m[0].queryIdx
    img2_idx = m[0].trainIdx

    orgX,orgY = kp1[img1_idx].pt
    corrX,corrY= kp2[img2_idx].pt

    if((orgY,orgX) in firstOrgs):
        continue
    else:
        firstOrgs.append((orgY,orgX))
        firstCorrs.append((corrY,corrX))

orgs=np.ones((len(firstOrgs),3))
corrs = np.ones((len(firstCorrs),3))
currentOrg = np.ones(3)
currentCorr = np.ones(3)
for i in range(0,len(firstOrgs)):
    currentOrg[1] = firstOrgs[i][1]
    currentOrg[0] = firstOrgs[i][0]

    currentCorr[1] = firstCorrs[i][1]
    currentCorr[0] = firstCorrs[i][0]

    orgs[i] = currentOrg
    corrs[i] = currentCorr


N = 10000
i = 0
selectedOrg = np.zeros((4,3))
selectedCorr = np.zeros((4,3))
bestInliers = -1

while(i<N):
    rndm = random.sample(range(0,len(orgs)-1),4)
    #print(rndm)
    for k in range(0,len(rndm)):
        selectedOrg[k] = orgs[rndm[k]]
        selectedCorr[k] = corrs[rndm[k]]

    H = dlt.DLT(selectedOrg,selectedCorr)
    inlierOrg = []
    inlierCorr = []
    transformed = transform(H,orgs)

    nmbOfInliers = 0
    for trans in range(0,len(transformed)):
        for corr in range(0,len(corrs)):
            norm = LA.norm(transformed[trans]-corrs[corr])
            if(norm < 3):
                inlierOrg.append(orgs[corr])
                inlierCorr.append(corrs[corr])
                nmbOfInliers+=1
                break
    if(nmbOfInliers>bestInliers):
        bestH = np.copy(H)
        bestInliers = nmbOfInliers
        bestOrgs = inlierOrg
        bestCorr = inlierCorr
        N = -2 / np.log10(1-(bestInliers/len(orgs))**4)
    i+=1



bestInliersOrgHom = np.array(bestOrgs)
bestInliersCorrHom = np.array(bestCorr)
newHom = dlt.DLT(bestInliersOrgHom,bestInliersCorrHom)

transInliers = transform(newHom,bestInliersOrgHom)

while(True):
    nmbOfInliers = 0
    inlierOrg = []
    inlierCorr = []
    for trans in range(0,len(transInliers)):
        for corr in range(0,len(bestInliersCorrHom)):
            norm = LA.norm(transInliers[trans]-bestInliersCorrHom[corr])
            if(norm < 3):
                inlierOrg.append(bestInliersOrgHom[corr])
                inlierCorr.append(bestInliersCorrHom[corr])
                nmbOfInliers+=1
                break
    if(nmbOfInliers <= bestInliers ):
        if(nmbOfInliers == bestInliers):
            break
        bestInliers = nmbOfInliers

    inlierOrgHom = np.array(inlierOrg)
    inlierCorrHom = np.array(inlierCorr)
    homog = dlt.DLT(inlierOrgHom,inlierCorrHom)
    j=0
    for point in bestInliersOrgHom:
        transformed[j] = homog @ point
        transformed[j][0] /= transformed[j][2]
        transformed[j][1] /= transformed[j][2]
        transformed[j][2] /= transformed[j][2]
        j+=1

x1 = np.dot(newHom, [160, 160, 1])
x2 = np.dot(newHom, [160, 464, 1])
x3 = np.dot(newHom, [575, 171, 1])
x4 = np.dot(newHom, [575, 450, 1])

a1x = int(x3[1] / x3[2])  # leftmost corner
a1y = int(x3[0] / x3[2])
b1x = int(x2[1] / x2[2])  # rightbottom corner
b1y = int(x2[0] / x2[2])

img3 = cv2.imread('img2.jpg')
img3 = cv2.rectangle(img3, (a1x, a1y), (b1x, b1y), (0, 0, 255), 2)  # if points are valid, draw rectangle

cv2.imwrite("final_image.png", img3)
