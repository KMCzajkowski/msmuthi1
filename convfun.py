import numpy as np

def listtoarr(A):
    return np.vstack((np.hstack((A[0],A[1])), np.hstack((A[2],A[3]))))


def blockconv(A):
    blk_size=(3,3)
    blk_sqrtlen=4
    blk_s2len=2

    b=0*A
    counter=0
    Aee=[]
    Aem=[]
    Ame=[]
    Amm=[]
    for ii in range(blk_s2len):
        for jj in range(blk_s2len):
            Ablk=A[6*ii:6*(ii+1),6*jj:6*(jj+1)]
            Aee.append(Ablk[0:3, 0:3])
            Aem.append(Ablk[0:3, 3:6])
            Ame.append(Ablk[3:6,0:3])
            Amm.append(Ablk[3:6, 3:6])
    B=[listtoarr(Aee), listtoarr(Aem), listtoarr(Ame), listtoarr(Amm)]
    B=listtoarr(B)
    return B


def blockdiag(M):
    zer=np.zeros(M.shape)
    Mb=[M, zer, zer, M]
    return listtoarr(Mb)


def blockflip(M):
    blocksize=3
    nblocks=2
    blocks=[]
    for ii in range(nblocks):
        for jj in range(nblocks):
            blocks.append(M[3*ii:3*(ii+1), 3*jj:3*(jj+1)])
    blocks.reverse()
    return listtoarr(blocks)
