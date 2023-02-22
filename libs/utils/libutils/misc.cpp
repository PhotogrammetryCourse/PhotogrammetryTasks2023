#include "misc.h"

#ifdef CUDA_SUPPORT
#include <cuda_runtime_api.h>
#endif

void gpu::printDeviceInfo(gpu::Device &device)
{
#ifdef CUDA_SUPPORT
	if (device.supports_cuda) {
		int driverVersion = 239;
		cudaDriverGetVersion(&driverVersion);
		std::cout << "GPU. " << device.name << " (CUDA " << driverVersion << ").";
	} else
#endif
	{
		ocl::DeviceInfo info;
		info.init(device.device_id_opencl);
		if (info.device_type == CL_DEVICE_TYPE_GPU) {
			std::cout << "GPU.";
		} else if (info.device_type == CL_DEVICE_TYPE_CPU) {
			std::cout << "CPU.";
		} else {
			throw std::runtime_error(
					"Only CPU and GPU supported! But type=" + to_string(info.device_type) + " encountered!");
		}
		std::cout << " " << info.device_name << ".";
		if (info.device_type == CL_DEVICE_TYPE_CPU) {
			std::cout << " " << info.vendor_name << ".";
		}
	}

	if (device.supportsFreeMemoryQuery()) {
		std::cout << " Free memory: " << (device.getFreeMemory() >> 20) << "/" << (device.mem_size >> 20) << " Mb";
	} else {
		std::cout << " Total memory: " << (device.mem_size >> 20) << " Mb";
	}
	std::cout << std::endl;
}


gpu::Device gpu::chooseDevice(bool verbose)
{
	std::vector <gpu::Device> devices = gpu::enumDevices();
	unsigned int device_index = std::numeric_limits<unsigned int>::max();

	if (devices.size() == 0) {
		gpu::Device cpu_device;
		cpu_device.supports_opencl = false;
		cpu_device.supports_cuda   = false;
		cpu_device.is_cpu          = true;
		return cpu_device;
	} else {
		if (verbose) {
			std::cout << "OpenCL devices:" << std::endl;
			for (int i = 0; i < devices.size(); ++i) {
				std::cout << "  Device #" << i << ": ";
				gpu::printDeviceInfo(devices[i]);
			}
		}
		{
			device_index = 0;

			// trying to find GPU (for cases when user has both GPU and CPU available as OpenCL/CUDA device)
			for (size_t i = 0; i < devices.size(); ++i) {
				if (!devices[i].is_cpu) {
					device_index = i;
					break;
				}
			}
		}
		if (verbose) {
			std::cout << "Using device #" << device_index << ": ";
			gpu::printDeviceInfo(devices[device_index]);
		}
	}
	return devices[device_index];
}
