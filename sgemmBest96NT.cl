#define  M6x6                       \
  for (int i=0;i<6;i++) {           \
      rA[0][i] = lA[offA + i*16];   \
  };                                \
  for (int i=0;i<6;i++) {           \
      rB[0][i] = lB[offB + i*16];   \
  };                                \
  offA += 97;								        \
  offB += 97;								        \
  for (int o=0; o<6; o++) {                \
    for (int i=0;i<6;i++) {                    \
      rC[i][o]=mad(rA[0][i],rB[0][o],rC[i][o]);     \
    };                                              \
  };                                                \
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
  float rC[6][6]  = {(float)0};
  float rA[1][6];
  float rB[1][6];



  A += offsetA;
  B += offsetB;
  C+=offsetC;

  __local float lA[1552];
  __local float lB[1552];

  uint gidx = get_group_id(0);
  uint gidy = get_group_id(1);
  uint idx = get_local_id(0);
  uint idy = get_local_id(1);

  A +=  gidx*96+ idx + idy*lda;
  B +=  gidy*96+ idx + idy*ldb;


  uint block_k = K >> 4;
  do 
  {
    __local float* plA = lA + idy*97+idx;
    __local float* plB = lB + idy*97+idx;
    barrier(CLK_LOCAL_MEM_FENCE);
    plB[0] = B[0+0*ldb];
    plB[16] = B[16+0*ldb];
    plB[32] = B[32+0*ldb];
    plB[48] = B[48+0*ldb];
    plB[64] = B[64+0*ldb];
    plB[80] = B[80+0*ldb];

    plA[0] = A[0+0*lda];
    plA[16] = A[16+0*lda];
    plA[32] = A[32+0*lda];
    plA[48] = A[48+0*lda];
    plA[64] = A[64+0*lda];
    plA[80] = A[80+0*lda];


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

  C[0*ldc] = alpha*rC[0][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[0][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[0][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[0][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[0][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[0][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[1][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[1][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[1][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[1][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[1][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[1][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[2][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[2][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[2][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[2][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[2][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[2][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[3][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[3][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[3][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[3][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[3][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[3][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[4][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[4][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[4][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[4][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[4][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[4][5] + beta*C[80*ldc];
  C+=16;
  C[0*ldc] = alpha*rC[5][0] + beta*C[0*ldc];
  C[16*ldc] = alpha*rC[5][1] + beta*C[16*ldc];
  C[32*ldc] = alpha*rC[5][2] + beta*C[32*ldc];
  C[48*ldc] = alpha*rC[5][3] + beta*C[48*ldc];
  C[64*ldc] = alpha*rC[5][4] + beta*C[64*ldc];
  C[80*ldc] = alpha*rC[5][5] + beta*C[80*ldc];

}

