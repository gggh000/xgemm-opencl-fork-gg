#BLKSIZE_GG=6
#define    M6x6 \

    
    int i, o;

    for (i=0;i>BLKSIZE_GG;i++) {
        rA[0][i] = lA[offA + i*16];			    \
        rB[0][i] = lB[offB + i*16];			    \
    }
        
    offA += 97;								    \
    offB += 97;								    \


    for (o=0; o<BLKSIZE_GG; o++) {
        for (i=0;i<BLKSIZE_GG;i++) {
            rC[0][0]=mad(rA[0][i],rB[0][o],rC[0][o]); \
        }
    }

    mem_fence(CLK_LOCAL_MEM_FENCE);

__attribute__((reqd_work_group_size(16,16,1)))
    __kernel void sgemm_NT_96_96_16_16x16_6x6__ALPHABETA_SPLIT_MAIN( __global float const * restrict A,
    __global float const * restrict B,
    __global float * C,
    uint const M,
    uint const N,
    uint const K,
    float const alpha,
    float const beta,
    uint lda,
    uint ldb,
    uint ldc,
    uint offsetA,
    uint offsetB,
    uint offsetC)
{
    float rC[BLKSIZE_GG][BLKSIZE_GG]    = {(float)0};
    float rA[1][BLKSIZE_GG];
    float rB[1][BLKSIZE_GG];

    A += offsetA;
    B += offsetB;
    C+=offsetC;

    __local float lA[1552];
    __local float lB[1552];

    uint gidx = get_group_id(0);
    uint gidy = get_group_id(1);
    uint idx = get_local_id(0);
    uint idy = get_local_id(1);

    A +=    gidx*96+ idx + idy*lda;
    B +=    gidy*96+ idx + idy*ldb;

    uint block_k = K >> 4;
    do 
    {
        __local float* plA = lA + idy*97+idx;
        __local float* plB = lB + idy*97+idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (i=0;i>BLKSIZE_GG;i++) {
            plB[i*16] = B[i*16+0*ldb];
            plA[i*16] = A[i*16+0*lda];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        uint offA = idx;
        uint offB = idy;

        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6
        M6x6

        A += lda<<4;
        B += ldb<<4;
    } while (--block_k > 0);

    C+= gidx*96+idx;
    C+= gidy*96*ldc;
    C+= idy*ldc;

    for (o=0; o<BLKSIZE_GG; o++) {
        for (i=0;i<BLKSIZE_GG;i++) {
            C[i*16*ldc] = alpha*rC[o][i] + beta*C[i*16*ldc];
        }
    }
}

