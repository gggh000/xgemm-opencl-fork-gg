//#include "ifort64/include/acml.h"
//#define NOCPU
#ifndef NOCPU
extern "C" {
#include "cblas.h"
}
#endif
#include <cstdio>

#include <cmath>

#include <sstream>
#include <cstring>
#include <string>

#include <iostream>
#include <fstream>

#include <vector>
#include "Timer.h"

#include "CL/opencl.h"
#include "threadManagement.h"

using namespace std;




  //const char* 
  //  getOpenCLErrorCodeStr(string input)
  //{
  //  return "unknown error code"; 
  //}





struct thread_data
{
  cl_context ctx;
  cl_command_queue queue;
  bool sgemm;
  bool mn;
  bool k;
  cl_kernel xgemmKernel;
  std::ofstream outputfile;
} ;

std::vector<cl_command_queue> queuelist;
std::vector <Thread> g_threads;
thread_data* g_data = NULL;

#define EPSILOND 10e-4
#define EPSILONF 10e-3
//Benchmark sizes
static const unsigned int size_min = 96*10;
static const unsigned int size_max = 7000;
static const unsigned int size_inc = 96;




std::size_t ntests = 1;



const char*  getOpenCLErrorCodeStr(int errorCode)
{

  switch(errorCode)
  {
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";               
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";           
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";      
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";                    
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";                 
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";        
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";                    
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";               
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";         
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";              
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";                         
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";                      
  case CL_INVALID_DEVICE_TYPE:
    return "CL_INVALID_DEVICE_TYPE";               
  case CL_INVALID_PLATFORM:
    return "CL_INVALID_PLATFORM";                   
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";                    
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";                    
  case CL_INVALID_QUEUE_PROPERTIES:
    return "CL_INVALID_QUEUE_PROPERTIES";           
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";              
  case CL_INVALID_HOST_PTR:
    return "CL_INVALID_HOST_PTR";                   
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";                  
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";    
  case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";                 
  case CL_INVALID_SAMPLER:
    return "CL_INVALID_SAMPLER";                    
  case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";                     
  case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";              
  case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";                    
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "CL_INVALID_PROGRAM_EXECUTABLE";          
  case CL_INVALID_KERNEL_NAME:
    return "CL_INVALID_KERNEL_NAME";                
  case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";          
  case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";                     
  case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX";                   
  case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE";                   
  case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE";                    
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";                
  case CL_INVALID_WORK_DIMENSION:
    return "CL_INVALID_WORK_DIMENSION";              
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";             
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";             
  case CL_INVALID_GLOBAL_OFFSET:
    return "CL_INVALID_GLOBAL_OFFSET";              
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";             
  case CL_INVALID_EVENT:
    return "CL_INVALID_EVENT";                      
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";                 
  case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";                  
  case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";                 
  case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";                   
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";            
  case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
    return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case CL_PLATFORM_NOT_FOUND_KHR:
    return "CL_PLATFORM_NOT_FOUND_KHR";
    //case CL_INVALID_PROPERTY_EXT:
    //    return "CL_INVALID_PROPERTY_EXT";
  case CL_DEVICE_PARTITION_FAILED_EXT:
    return "CL_DEVICE_PARTITION_FAILED_EXT";
  case CL_INVALID_PARTITION_COUNT_EXT:
    return "CL_INVALID_PARTITION_COUNT_EXT"; 
  default:
    return "unknown error code";
  }
}

std::string get_file_contents(const char *filename)
{
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (in)
  {
    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return contents;
  }
  throw new std::string("File not found!");
}

inline double gflops(std::size_t M, std::size_t N, std::size_t K, double tsec){
  return 2.0*M*N*K*1e-9/tsec;
}

inline void check_err(int err, const char * fctname, cl_context* ctx)
{
  if (err != CL_SUCCESS)
  {
    const char* error=getOpenCLErrorCodeStr(err);

    printf( "%s failed with error : %s\n", fctname, error );
    if(ctx)
      clReleaseContext(*ctx);
    exit(EXIT_FAILURE);
  }
}


void cpu_gemm(size_t M, size_t N, size_t K, double alpha, double* A, size_t lda, double* B, size_t ldb, double beta, double* C, size_t ldc)
{	
  for(std::size_t j=0 ; j<N; ++j){
    for(std::size_t i=0 ; i<M; ++i){
      double rC = C[i + j*ldc];
      double prod = 0;
      for(std::size_t k=0 ; k<K; ++k){
        double rA = A[i + k*lda];
        double rB = B[j + k*ldb];
        prod += rA*rB;
      }
      C[i + j*ldc] = alpha*prod + beta*rC;
    }
  }

}

#ifndef NOCPU

bool checkResultCPU(double* DeviceResult, double* A, double* B,double* RefC, double alpha, double beta, int K, int M, int N, int lda,int ldb, int ldc)
{
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, RefC, ldc);

  for ( int i = 0; i< M*N; i++)
  {
    if (abs(DeviceResult[i]-RefC[i])>=EPSILOND)
      return false;
  }

  return true;
}


bool checkResultCPU(float* DeviceResult, float* A, float* B,float* RefC, float alpha, float beta, int K, int M, int N, int lda,int ldb, int ldc)
{
  //cout<<  DeviceResult[0]<<endl;
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, RefC, ldc);

  for ( int i = 0; i< M*N; i++)
  {
    if (abs(DeviceResult[i]-RefC[i])>=EPSILONF)
    {
      printf("sgemm fails, i=%i, diff=%f\n", i, DeviceResult[i]-RefC[i]);
      printf("sgemm fails, i=%i, DeviceResult=%f, RefC=%f.\n", i, DeviceResult[i], RefC[i]);
      return false;
    //} else {
      //printf("sgemm ok, i=%i, diff=%f\n", i, DeviceResult[i]-RefC[i]);
      //printf("sgemm ok, i=%i, DeviceResult=%f, RefC=%f.\n", i, DeviceResult[i], RefC[i]);
    }
  }

  return true;
}
#endif

template <typename NumericT>
double run(cl_context ctx, cl_command_queue queue, cl_kernel kernel, size_t M, size_t N, size_t K, bool* test)
{
  cl_uint clM = (cl_uint)M;
  cl_uint clN = (cl_uint)N;
  cl_uint clK = (cl_uint)K;

  int lda = (int)M;
  int ldb = (int)N;
  int ldc = (int)M;

  cl_int err;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  NumericT alpha = 1.0;
  NumericT beta = 0.0;
  cl_uint offset = 0;


  NumericT* A = new NumericT[M*K];
  NumericT* B = new NumericT[N*K];
  NumericT* C = new NumericT[M*N];
  NumericT* gpuC = new NumericT[M*N];



  if(test){
    for (std::size_t i=0; i<M*K; ++i)
      A[i] = (NumericT)rand()/RAND_MAX;

    for (std::size_t i=0; i<K*N; ++i)
      B[i] = (NumericT)rand()/RAND_MAX;

    for(std::size_t i=0 ; i<M*N; ++i)
      C[i] = (NumericT)rand()/RAND_MAX;
  }

  /* Prepare OpenCL memory objects and place matrices inside them. */
  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M*K * sizeof(NumericT), NULL, &err);
  check_err(err, "clCreateBuffer A", &ctx);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K*N * sizeof(NumericT), NULL, &err);
  check_err(err, "clCreateBuffer B", &ctx);

  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M*N * sizeof(NumericT), NULL, &err);
  check_err(err, "clCreateBuffer C", &ctx);

  err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M * K * sizeof(NumericT), &(A[0]), 0, NULL, NULL);
  check_err(err, "clEnqueueWriteBuffer A", &ctx);
  err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K * N * sizeof(NumericT), &(B[0]), 0, NULL, NULL);
  check_err(err, "clEnqueueWriteBuffer B", &ctx);
  err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M * N * sizeof(NumericT), &(C[0]), 0, NULL, NULL);
  check_err(err, "clEnqueueWriteBuffer C", &ctx);


  std::size_t BlockSize[2];
  std::size_t ls[2]={16,16};



  BlockSize[0]=6;
  BlockSize[1]=6;
  ls[0] = ls[0]/(sizeof(NumericT)/4);
  ls[1] = ls[1]/(sizeof(NumericT)/4);



  std::size_t GlobalX = ((M+BlockSize[0] -1)/BlockSize[0]) ;
  GlobalX = ((GlobalX + ls[0]-1)/ls[0])*ls[0] ;

  std::size_t GlobalY = ((N+BlockSize[1] - 1)/BlockSize[1]) ;
  GlobalY = ((GlobalY + ls[1]-1)/ls[1])*ls[1] ;

  std::size_t gs[2] = {GlobalX, GlobalY};

  int ArgIndex = 0;



  cl_int error = clSetKernelArg(kernel, ArgIndex++, sizeof(cl_mem), &bufA);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_mem), &bufB);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_mem), &bufC); 
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_uint), &clM);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_uint), &clN);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_uint), &clK);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(NumericT), &alpha); 
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(NumericT), &beta);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_int), &lda); 
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_int), &ldb);  
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_int), &ldc);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_uint), &offset);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_uint), &offset);
  error |= clSetKernelArg(kernel, ArgIndex++, sizeof(cl_uint), &offset);

  if(error!=CL_SUCCESS)
  {
    cout<< " FAILED : error setting Arguments"<<endl;
    getOpenCLErrorCodeStr(error);
    return 0.0;
  }

  // std::cout<<"B[352] = "<<B[352]<<std::endl;
#ifndef NOCPU
  std::cout  << "GG: NOCPU (GPU)";
  if(test)
  {
    // cpu_gemm(M,N,K,alpha,&(A[0]),M,&(B[0]),N,beta,&(C[0]),M);
    error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gs, ls, 0, NULL, NULL);
    if(error!=CL_SUCCESS)
    {
      cout<< " FAILED : clEnqueueNDRangeKernel"<<endl;
      getOpenCLErrorCodeStr(error);
      return 0.0;
    }
    error = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M * N * sizeof(NumericT), &(gpuC[0]), 0, NULL, NULL);

    if(error!=CL_SUCCESS)
    {
      cout<< " FAILED : clEnqueueReadBuffer"<<endl;
      getOpenCLErrorCodeStr(error);
      return 0.0;
    }

    *test = checkResultCPU(gpuC, A, B, C,  alpha,  beta, (int) K, (int) M, (int) N, (int) lda, (int)ldb, (int) ldc);

    if(*test==false)
    {
      std::cout << "Test M = " << M<<", N =  "<< N<<", K =  "<< K<< " failed!" << std::endl;
      exit(EXIT_FAILURE);
    }
    else
    {
      std::cout << "Test M = " << M<<", N =  "<< N<<", K =  "<< K<< " passed!" << std::endl;
    }
  }

#endif

  // Warm-up
  for (int i = 0; i<10; i++)
    error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gs, ls, 0, NULL, NULL);
  //error = clFinish(queue);
  clEnqueueMarkerWithWaitList(queue, 0, NULL, NULL);


  // Benchmark
  cl_ulong start; 
  cl_ulong end;
  error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gs, ls, 0, NULL, &event);
  error = clFinish(queue);

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

  double time = (double)(end - start)*1e-9;

  // Test

  CPerfCounter timer;
  timer.Start();
  error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gs, ls, 0, NULL, NULL);
  error = clFinish(queue);
  timer.Stop();
  double APItime = timer.GetElapsedTime();
  std::cout  << "Performance xgemm API : " << gflops(M, N, K, APItime) << " GFlops, "<</*timeAPI << " s" <<  std::endl<<*/ std::endl;


  /* Release OpenCL memory objects. */
  delete [] A;
  delete [] B;
  delete [] C;
  delete [] gpuC;

  clReleaseMemObject(bufA);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufC);
  std::cout.flush();
  return time;
}



void TestProcedure(thread_data* data )
{
  bool passed = true;
  int offset = 0;


  size_t sizeKernelName = 0;
  char* KernelName = NULL;
  clGetKernelInfo(data->xgemmKernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &sizeKernelName);
  if (sizeKernelName)
  {
    KernelName = new char[sizeKernelName];
    clGetKernelInfo(data->xgemmKernel, CL_KERNEL_FUNCTION_NAME, sizeKernelName, KernelName, NULL);
  }

  double timeKernel = 0.0;
  double timeAPI = 0.0;

  size_t Inc = size_inc;
  if(!data->sgemm)
    Inc/=2;

  for (std::size_t size = size_min; size <= size_max; size +=Inc)
  {
    if(data->sgemm)
      timeKernel = run<float>( data->ctx, data->queue, data->xgemmKernel, size, size, size, &passed);
    else
      timeKernel = run<double>( data->ctx, data->queue, data->xgemmKernel, size, size, size, &passed);


    std::cout << "perf kernel : " << KernelName<<endl;
    std::cout<<size<<","<<size<<","<<size<<" : "  << gflops(size, size, size, timeKernel) << " GFLOPS, time to execute kernel : "<< timeKernel*1000 <<std::endl<<std::endl;

    data->outputfile<<size<<","<<size<<","<<size<<","<<","<< gflops(size, size, size, timeKernel)<<std::endl;

  }
  if (KernelName)
    delete [] KernelName;


}

void Print_BuildLog(cl_program program)
{

  for (size_t i=0; i<g_threads.size(); i++)
  {
    cl_device_id device;
    clGetCommandQueueInfo( 	g_data[i].queue,CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);


    //here we know program exist
    size_t NBChar = 0; 
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &NBChar);

    char* log = new char[NBChar];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NBChar, log, &NBChar);
    std::cout <<log<<std::endl;
    std::cout.flush();

    delete [] log;
  }
}


cl_int FinCLPlatform(cl_platform_id& platform)
{
  cl_int status = CL_SUCCESS;
  cl_uint numPlatforms;
  //cl_platform_id platform = NULL;
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if(status != CL_SUCCESS)
  {
    cout<<"Error: clGetPlatformIDs failed. Error code : "<< status;

    return status;
  }

  if (0 < numPlatforms)
  {
    // Get selected platform
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if(status != CL_SUCCESS)
    {
      cout<<"Error: clGetPlatformIDs failed. Error code : " <<status;

      return status;
    }

    // Print all platforms
    for (unsigned i = 0; i < numPlatforms; ++i)
    {
      char pbuf[100];
      status = clGetPlatformInfo(platforms[i],
        CL_PLATFORM_VENDOR,
        sizeof(pbuf),
        pbuf,
        NULL);

      if(status != CL_SUCCESS)
      {
        cout<<"Error: clGetPlatformInfo failed. Error code : ";
        return status;
      }

      cout << "Platform " << i << " : " << pbuf << endl;
    }

    // Get AMD platform
    for (unsigned i = 0; i < numPlatforms; ++i)
    {
      char pbuf[100];
      status = clGetPlatformInfo(platforms[i],
        CL_PLATFORM_VENDOR,
        sizeof(pbuf),
        pbuf,
        NULL);

      if(status != CL_SUCCESS)
      {
        cout<<"Error: clGetPlatformInfo failed. Error code : ";
        return status;
      }

      platform = platforms[i];
      if (!strcmp(pbuf, "Advanced Micro Devices, Inc."))
      {
        break;
      }
    }

    // Check for AMD platform
    char pbuf[100];
    status = clGetPlatformInfo(platform,
      CL_PLATFORM_VENDOR,
      sizeof(pbuf),
      pbuf,
      NULL);

    if(status != CL_SUCCESS)
    {
      cout<<"Error: clGetPlatformInfo failed. Error code : ";
      return status;
    }
    if (strcmp(pbuf, "Advanced Micro Devices, Inc."))
    {
      cout << "AMD platform not found" << endl;
      return -1;
    }

  }

  return status;

}


/*************************************************************************************************/

cl_int InitCL( std::vector<unsigned int>& deviceNum, bool deviceAll, cl_context& context)
{
  cl_int status = CL_SUCCESS;
  cl_platform_id platform = NULL;
  cl_context_properties properties[3];


  status = FinCLPlatform(platform);

  if(status!=CL_SUCCESS || platform==NULL)
  {
    cout<< "can't find a AMD platform for OpenCL" << endl;
    return status;
  }

  //I only want to test on GPU for the moment,
  unsigned int NBDevices = 0;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &NBDevices);

  if (NBDevices<1)
  {
    cout<< "no AMD GPU devices in the system or can't query them"<<endl;
    return -1;
  }

  //we test on all devices
  unsigned int MaxDevice = (unsigned int)deviceNum.size();
  int NBDeviceToUse =0;
  if (deviceAll)
    NBDeviceToUse = NBDevices;
  else
    NBDeviceToUse = NBDevices<=MaxDevice?NBDevices:MaxDevice;
  
  cl_device_id* devices = new cl_device_id[NBDevices];
  cl_device_id* devicesToUse = NULL;



  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, NBDevices, devices, NULL);
  if(status!=CL_SUCCESS)
  {

    return status;
  }


  
  if (deviceAll)
    devicesToUse = devices;
  else
  {
    devicesToUse =  new cl_device_id[MaxDevice];
    for(unsigned int i =0; i<MaxDevice; i++)
    {
      if (deviceNum[i]<NBDevices)
        devicesToUse[i]=devices[deviceNum[i]];
    }
  }

   // context properties list - must be terminated with 0
  properties[0]= CL_CONTEXT_PLATFORM;
  properties[1]= (cl_context_properties) platform;
  properties[2]= 0;

  // create a context with the GPU device
  context = clCreateContext(properties,NBDeviceToUse,devicesToUse,NULL,NULL,&status);
  if(status!=CL_SUCCESS)
  {

    return status;
  }


  cl_queue_properties QueueProp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
  queuelist.resize(NBDeviceToUse);
  for ( int i=0; i<NBDeviceToUse; i++)
  {
    queuelist[i] = clCreateCommandQueueWithProperties(context, devicesToUse[i], QueueProp, &status);

    if(status!=CL_SUCCESS)
    {
      return status;
    }

  }

  delete []devices;
  if (!deviceAll)
    delete []devicesToUse;


  return status;
}



//We will work only with clBLAS column major kernel
void createThread(bool sgemm, /*const clblasOrder Order,*/  cl_context context, cl_program programXgemm)
{
  g_threads.resize(queuelist.size());
  g_data = new thread_data[queuelist.size()];
  
  
  

  for (size_t i=0; i<g_threads.size(); i++)
  {
    stringstream ss;
    cl_int err = CL_SUCCESS;
    cl_kernel kernelXgemm;
    if (sgemm)
      kernelXgemm = clCreateKernel(programXgemm,"sgemm_NT_96_96_16_16x16_6x6__ALPHABETA_SPLIT_MAIN",&err);
    else
      kernelXgemm = clCreateKernel(programXgemm,"dgemm_NT_48_48_8_8x8_6x6__ALPHABETA_SPLIT_MAIN",&err);
    
    check_err(err, "clCreateKernel", &context);

    ss<<i;
    string cWaveSize = ss.str(); 

    string fileName="PerfoGemm_GPU_" + cWaveSize + ".csv";
    g_data[i].queue=queuelist[i];
    g_data[i].outputfile.open(fileName.c_str());
    g_data[i].outputfile<<"M"<<","<<"N"<<","<<"K"<<","<<"PerfAPI"<<std::endl;

    g_data[i].ctx = context;
    g_data[i].mn = false;
    g_data[i].k = false;
    g_data[i].sgemm = sgemm;
    g_data[i].xgemmKernel = kernelXgemm;

    
    //g_data.push_back(basicInput);

    g_threads[i].create((THREAD_PROC)TestProcedure,  (void*)&g_data[i]);

  }
}

int main( int argc, char *argv[])
{
  cl_int err;
  cl_context ctx = 0;
  int ret = 0;



  bool sgemm=true;
  bool deviceAll = true;
  std::vector<unsigned int> deviceNum;



  while (--argc)
  {
    ++argv;
    if (!strncmp(*argv, "S", 1)) 
      sgemm = true;
    else if (!strncmp(*argv, "D", 1)) 
      sgemm = false;
    else if (!strncmp(*argv, "iDD=", 4))
    {
      char *p = *argv+4;
      while (p) 
      {
        deviceAll = false;
        int device = 0;
        sscanf(p+1, "%d",&device);
        deviceNum.push_back(device);
        p = strchr(p, ',');
      }
    }
  }

  InitCL(deviceNum, deviceAll, ctx);

  if (sgemm)
    cout<<"this program will run sgemm"<<endl;
  else
    cout<<"this program will run dgemm"<<endl;

  /* Create and build program */
  std::string src;


  //sgemm1
  std::string srcXgemm;
  if (sgemm)
    srcXgemm = get_file_contents("sgemmBest96NT.cl");
  else
    srcXgemm = get_file_contents("dgemmBest48NT.cl");

  const char * csrcXgemm = srcXgemm.c_str();
  std::size_t srcsizeXgemm[] = {strlen(csrcXgemm)};



  cl_program programXgemm = clCreateProgramWithSource(ctx,1,&csrcXgemm,srcsizeXgemm,&err);
  check_err(err, "clCreateProgramWithSource", &ctx);

  err = clBuildProgram(programXgemm, 0, NULL, "-cl-std=CL2.0 -Wb,-hsail-reg-slots=8 -Wb,-hsail-reg64-pressure-limit=44 -fsc-use-buffer-for-hsa-global "  , NULL, NULL);
  //check_err(err, "clCreateProgramWithSource", &ctx);
  if (err!=CL_SUCCESS)
    Print_BuildLog(programXgemm);




  //bool passed;


  // Benchmark
  std::cout << "Benchmarking square matrices..." << std::endl;
  std::cout << "-------------------" << std::endl;

  createThread(sgemm,  ctx, programXgemm);

  for (int i=0;i<g_threads.size();i++)	  
  {
    g_threads[i].join();
  }


  for (size_t i=0; i<g_threads.size(); i++)
  {   
    clReleaseCommandQueue(g_data[i].queue);
    clReleaseKernel(g_data[i].xgemmKernel);
  }
  delete [] g_data;

  clReleaseProgram(programXgemm);

  clReleaseContext(ctx);

  return ret;
}






