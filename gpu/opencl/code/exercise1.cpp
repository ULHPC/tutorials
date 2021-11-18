
//Original tutorial: "Tutorial: Simple start with OpenCL and C++", 
//https://programmerclick.com/article/47811146604/


#define SIZE 10
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <iostream>

using namespace std; 


// kernel calculates for each element C=A+B
std::string kernel_code =
    "   void kernel simple_add(global const int* A, global const int* B, global int* C){ "
    "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
    "   }                                                                               ";

int main() {

    //If there are no opencl platforms -  all_platforms == 0 and the program exits. 

    //One of the key features of OpenCL is its portability. So, for instance, there might be situations
    // in which both the CPU and the GPU can run OpenCL code. Thus, 
    // a good practice is to verify the OpenCL platforms to choose on which the compiled code run.

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No OpenCL platforms found.\n";
        exit(1);
    }

    //We are going to use the platform of id == 0
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";


    //An OpenCL platform might have several devices. 
    //The next step is to ensure that the code will run on the first device of the platform, 
    //if found. 

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context({ default_device });


    // create buffers on the device
    cl::Buffer A_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    cl::Buffer B_d(context, CL_MEM_READ_ONLY, sizeof(int) * SIZE);
    cl::Buffer C_d(context, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE);

    int A_h[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int B_h[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

    //create queue to push commands to the device.
    cl::CommandQueue queue(context, default_device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(A_d, CL_TRUE, 0, sizeof(int) * SIZE, A_h);
    queue.enqueueWriteBuffer(B_d, CL_TRUE, 0, sizeof(int) * SIZE, B_h);

    cl::Program::Sources sources;

    //Appending the kernel, which is presented here as a string. 
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    //OpenCL compiles the kernel in runtime, that's the reason it is expressed as a string. 
    //There are also ways to compile the device-side code offline. 
    cl::Program program(context, sources);


    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }
    //If runtime compilation are found they are presented in this point of the program.


    //From the program, which contains the "simple_add" kernel, create a kernel for execution
    //with three cl:buffers as parameters.
    //The types must match the arguments of the kernel function. 
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> simple_add(cl::Kernel(program, "simple_add"));
    
    //Details to enqueue the kernel for execution.
    cl::NDRange global(SIZE);
    simple_add(cl::EnqueueArgs(queue, global), A_d, B_d, C_d).wait();

    int C_h[SIZE];
    //read result C_d from the device to array C_h
    queue.enqueueReadBuffer(C_d, CL_TRUE, 0, sizeof(int) * SIZE, C_h);

    std::cout << " result: \n";
    for (int i = 0; i<10; i++) {
        std::cout << C_h[i] << " ";
    }

    return 0;
}