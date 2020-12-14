/*
 * Example macros for error handling.
 */

#ifndef __HELPERS_H__
	#define __HELPERS_H__

	#define CUDIE(result) { \
		cudaError_t e = (result); \
		if (e != cudaSuccess) { \
			std::cerr << __FILE__ << ":" << __LINE__; \
			std::cerr << " CUDA runtime error: " << cudaGetErrorString(e) << '\n'; \
			exit((int)e); \
		}}

	#define CUDIE0() CUDIE(cudaGetLastError())
#endif

