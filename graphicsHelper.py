import numpy as np 
import trimesh 
import trimesh.ray.ray_pyembree as rayMeshInt
import scipy.interpolate
from scipy import ndimage
import cv2
from scipy.sparse import csr_matrix
import pdb

def PixelCoordToWorldCoord(K, R, t, u, v, depth):
    a = np.multiply(K[2,0], u) - K[0,0]
    b = np.multiply(K[2,1], u) - K[0,1]
    c = np.multiply(depth, (K[0,2] - np.multiply(K[2,2], u)))

    g = np.multiply(K[2,0], v) - K[0,1]
    h = np.multiply(K[2,1], v) - K[1,1]
    l = np.multiply(depth, (K[1,2] - np.multiply(K[2,2], v)))

    y = np.divide(l - np.divide(np.multiply(g, c), a), h - np.divide(np.multiply(g, b), a))
    x = np.divide((c - np.multiply(b,y)), a)
    z = depth
    C = np.concatenate(([x], [y], [z]), axis = 0)
    W = np.matmul(R.T, C - t)
    return W.T

def WorldCoordTopixelCoord(K, R, t, W):
    C = t + np.matmul(R, W.T)
    p = np.matmul(K, C)
    Xp = np.divide(p[0:1, :], p[2:3, :])
    Yp = np.divide(p[1:2, :], p[2:3, :])
    return Xp, Yp

class Camera:
    def __init__(self, name, K, R, t, image):
        self.name = name
        self.K = K
        self.R = R
        self.t = t
        self.image = image
        x = np.linspace(0, image.shape[1], image.shape[1], axis = -1)   
        y = np.linspace(0, image.shape[0], image.shape[0], axis = -1)
        xv, yv = np.meshgrid(x, y)
        depth = np.zeros_like(xv)
        self.C = PixelCoordToWorldCoord(K, R, t, [0], [0], [0])
        temp = PixelCoordToWorldCoord(K, R, t, xv.flatten(), yv.flatten(), depth.flatten() + 1)
        self.rays = temp - self.C
        self.opticalAxisRay = self.GetOpticalAxisRay()

    def GetIntersectionWithLocation(self, rayInt):
        loc, index_ray, index_tri = rayInt.intersects_location(np.tile(self.C, [self.rays.shape[0], 1]), self.rays, multiple_hits=False)
        return loc, index_ray, index_tri

    def GetOpticalAxisRay(self):
        return PixelCoordToWorldCoord(self.K, self.R, self.t, self.K[0, 2], self.K[1, 2], [1]) - self.C

    def GetImageAtPixelCoordsOld(self, X, Y, indices, image):
        imageInterpR = np.zeros((image.shape[0], image.shape[1]))
        imageInterpG = np.zeros((image.shape[0], image.shape[1]))
        imageInterpB = np.zeros((image.shape[0], image.shape[1]))
        validMaskR = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        imageInterpR.flat[indices] = ndimage.map_coordinates(image[:,:,0], [Y, X], order=1)
        imageInterpG.flat[indices] = ndimage.map_coordinates(image[:,:,1], [Y, X], order=1)
        imageInterpB.flat[indices] = ndimage.map_coordinates(image[:,:,2], [Y, X], order=1)
        imageInterp = np.stack((imageInterpR, imageInterpG, imageInterpB), axis = 2)
        validMaskR.flat[indices] = True
        validMask = np.stack((validMaskR, validMaskR, validMaskR), axis = 2)
        return imageInterp, validMask

    def GetImageAtPixelCoords(self, X, Y, indices, image):
        imageInterpAll = []
        validMask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        for c in range(image.shape[2]):
            imageInterp = np.zeros((image.shape[0], image.shape[1]))
            imageInterp.flat[indices] = ndimage.map_coordinates(image[:,:,c], [Y, X], order=1)
            imageInterpAll.append(imageInterp.copy())
        imageInterpAll = np.stack(imageInterpAll, axis = 2)
        validMask.flat[indices] = True
        validMask = np.tile(validMask[:,:,np.newaxis], (1,1,imageInterpAll.shape[2]))
        return imageInterp, validMask

    def GetImageAtPixelCoordsRisky(self, X, Y, image):
        imageInterpR = ndimage.map_coordinates(image[:,:,0], [Y, X], order=1)
        imageInterpG = ndimage.map_coordinates(image[:,:,1], [Y, X], order=1)
        imageInterpB = ndimage.map_coordinates(image[:,:,2], [Y, X], order=1)
        imageInterp = np.concatenate((imageInterpR, imageInterpG, imageInterpB), axis = 1) #TODO change 1 to 2 if X is image-like?
        return imageInterp

    def GetImageAtWorldCoords(self, W, WTriIndices, imageIndices, rayInt, imageToUse):
        rays = W - self.C
        WValidTriIndices = rayInt.intersects_first(np.tile(self.C, [rays.shape[0], 1]), rays)
        # Only those intersections are valid which occur at the same triangle as we wanted:
        validWorld = W[WValidTriIndices == WTriIndices, :]
        validXp, validYp = WorldCoordTopixelCoord(self.K, self.R, self.t, validWorld)
        imageValidIndices = imageIndices[WValidTriIndices == WTriIndices]
        image, validMask = self.GetImageAtPixelCoords(validXp, validYp, imageValidIndices, imageToUse)
        return image, validMask

    def GetValuesAtWorldCoords(self, W, rayInt, imageToUse):
        rays = W - self.C
        #TODO: Check if rays need to be rescaled
        locations, index_ray, _ = rayInt.intersects_location(np.tile(self.C, [rays.shape[0], 1]), rays, multiple_hits=False)
        validMask = np.sum(np.abs(W[index_ray] - locations), axis = 1) <= 1.0
        # Only those intersections are valid which occur at the same triangle as we wanted:
        Xp, Yp = WorldCoordTopixelCoord(self.K, self.R, self.t, locations)
        _, validVertexIndices = np.nonzero(np.logical_and(np.logical_and(np.logical_and(Xp >= 0, Yp >=0), np.logical_and(Xp <= imageToUse.shape[1] - 1, Yp <= imageToUse.shape[0] - 1)), validMask))
        validXp = Xp[:, validVertexIndices].T
        validYp = Yp[:, validVertexIndices].T
        imageValues = self.GetImageAtPixelCoordsRisky(validXp, validYp, imageToUse)
        return imageValues, validVertexIndices

    #TODO: Assumes that all world coordinates are visible!
    def GetImageAtWorldCoordsRisky(self, W, imageToUse):
        validXp, validYp = WorldCoordTopixelCoord(self.K, self.R, self.t, W)
        image = self.GetImageAtPixelCoordsRisky(validXp, validYp, imageToUse)
        return image

    def ComputeDepth(self, W):
        rays = W - self.C
        return np.abs(np.dot(rays, self.opticalAxisRay.T))

    def GetColorAtVertices(self, mesh, Wi, index_ray_i, image):
        distanceToVertex, affectedVertexIndices = trimesh.proximity.ProximityQuery(mesh).vertex(Wi)
        # uniqueVertices, indices, inverse = np.unique(affectedVertexIndices, return_index = True, return_inverse = True) #TODO: Better to find closest
        pair = np.zeros(affectedVertexIndices.shape[0], dtype=[('index', 'i4'), ('distance', 'f4')])
        pair['index'] = affectedVertexIndices
        pair['distance'] = distanceToVertex
        order = np.argsort(pair, order=['index', 'distance'])
        sortedPair = pair[order]
        sortedVertices = sortedPair['index']
        uniqueVertices, vertexIndicesSorted = np.unique(sortedVertices, return_index=True)
        uniqueOrder = order[vertexIndicesSorted]
        validImageVals = np.stack((image[:,:,0].flatten()[index_ray_i], image[:,:,1].flatten()[index_ray_i], image[:,:,2].flatten()[index_ray_i]), axis = 1)
        validImageValsVertices = validImageVals[uniqueOrder, :]
        pdb.set_trace()
        return validImageValsVertices, uniqueVertices

    def GetImageToVertexSmoothingMatrix(self, mesh, Wi, index_ray_i):
        M = mesh.vertices.shape[0]
        N = Wi.shape[0]
        distanceToVertex, affectedVertexIndices = trimesh.proximity.ProximityQuery(mesh).vertex(Wi)
        rowIndices = np.zeros((Wi.shape[0]))
        colIndices = np.zeros((Wi.shape[0]))
        weight = np.zeros((Wi.shape[0]))
        for w in range(N):
            rowIndices[w] = affectedVertexIndices[w]
            colIndices[w] = w # can be vectorized
            weight[w] = 1.0 / (1e-3 + distanceToVertex[w])
        mat = csr_matrix((weight, (rowIndices, colIndices)), shape=(M, N))
        normalizingFactor = mat.sum(axis = 1) + 1e-3
        return mat, normalizingFactor

    def GetVertexColorWithSmoothingMatrix(self, mat, normalizingFactor, image, index_ray_i):
        validImageVals = np.stack((image[:,:,0].flatten()[index_ray_i], image[:,:,1].flatten()[index_ray_i], image[:,:,2].flatten()[index_ray_i]), axis = 1)
        vertexValues = np.divide(mat * validImageVals, normalizingFactor)
        return vertexValues

def AlignNormals(N, ref):
    dotP = np.sum(np.multiply(N, ref), axis = 1)
    N[dotP < 0, :] = N[dotP < 0, :] * -1
    return N

def ComputeVertexNormals(mesh):
    faceNormals, faceNormalsValid = trimesh.triangles.normals(mesh.triangles)
    vertexNormals = np.zeros((mesh.vertices.shape[0], 3))
    visited = np.zeros((mesh.vertices.shape[0], 1))
    faceIndex = 0
    for currentFace in mesh.faces:
        if not faceNormalsValid[faceIndex]:
            continue
        currentFaceNormal = faceNormals[faceIndex]
        for vertex in currentFace:
            aligned = -1.0 if np.dot(vertexNormals[vertex, :], currentFaceNormal) < 0.0 else 1.0
            vertexNormals[vertex, :] = vertexNormals[vertex, :] + aligned * currentFaceNormal
            visited[vertex] = visited[vertex] + 1 
        faceIndex = faceIndex + 1
    
    normMagnitude = np.sqrt(np.sum(np.power(vertexNormals, 2.0), axis = 1)) + 1e-5
    vertexNormals[:,0] = np.divide(vertexNormals[:,0], normMagnitude)
    vertexNormals[:,1] = np.divide(vertexNormals[:,1], normMagnitude)
    vertexNormals[:,2] = np.divide(vertexNormals[:,2], normMagnitude)
    return vertexNormals

def Normalize01(a):
    b = (a - np.min(a))/np.ptp(a)
    return b
