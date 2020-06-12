This test requires OpenCL 2.0 compiler stack (HSAIL compiler stack)
Therefore it can only run on CL 2.0 compatible devices
To be sure it can only be run on such devcies, it uses clCreateCommandQueueWithProperties which is only declared on CL2.0 SDK.

You will need to install AMD APP SDK 3.0 or newer to get this API.
