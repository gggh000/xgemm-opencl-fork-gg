#define BLOCK_SZ 6
#define  M6x6                       \
  for (int i=0;i<BLOCK_SZ;i++) {           \
      rA[0][i] = lA[offA + i*16];   \
  };                                \
  for (int i=0;i<BLOCK_SZ;i++) {           \
      rB[0][i] = lB[offB + i*16];   \
  };                                \
  offA += 97;								        \
  offB += 97;								        \
  for (int o=0; o<BLOCK_SZ; o++) {                \
    for (int i=0;i<BLOCK_SZ;i++) {                    \
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
  float rC[BLOCK_SZ][BLOCK_SZ]  = {(float)0};
  float rA[1][BLOCK_SZ];
  float rB[1][BLOCK_SZ];

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

    for (int i=0;i<BLOCK_SZ; i++) {
      plB[i*16] = B[16*i+0*ldb];
    };

    for (int i=0;i<BLOCK_SZ; i++) {
      plA[i*16] = A[16*i+0*ldb];
    };

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

  for (int o = 0 ; o < BLOCK_SZ ; o++ ) {
    for (int i = 0 ; i < BLOCK_SZ ; i++) {
      C[16*i*ldc] = alpha*rC[o][i] + beta*C[i*16*ldc];
    };
    C+=16;
  };
}
