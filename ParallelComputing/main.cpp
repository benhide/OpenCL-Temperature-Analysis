#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <chrono>
#include "Utils.h"

// ******************************************************************************************************************************************************************
// *************************************************************************TYPE DEFINITIONS*************************************************************************
// ******************************************************************************************************************************************************************

typedef chrono::high_resolution_clock hi_res_clock;
typedef chrono::high_resolution_clock::time_point hi_res_time_point;
typedef int integer;
typedef float floating_point;

// ******************************************************************************************************************************************************************
// **************************************************************************GLOBAL VARIABLES************************************************************************
// ******************************************************************************************************************************************************************

// Mean values
float mean_float;
int mean_int;

// Variance
float variance_float;

// Used to convert  milli seconds to seconds
const float milli_to_seconds = 1000.0f;

// Number of data enteries
size_t number_of_data_entries;

// File directory / name -----> "temp_lincolnshire.txt" OR "temp_lincolnshire_short.txt"
const char* file = "temp_lincolnshire.txt"; 

// Device info
cl::Device device;
size_t prefferSize = 0;

// ******************************************************************************************************************************************************************
// ************************************************************************FUNCTION PROTOITYPES**********************************************************************
// ******************************************************************************************************************************************************************

// Print help
void print_help();

// Load file function
vector<floating_point> load_file_float(const char* file);

// Parse each line of the file
floating_point parse_string_to_float(string line);

// Load file function
vector<integer> load_file_int(const char* file);

// Parse each line of the file
integer parse_string_to_int(string line);

// *******************************************************************************FLOATS*****************************************************************************

// Floating point kernel calls
void floating_point_kernel_calls(size_t input_size, cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, vector<floating_point> air_temperatures, size_t local_size);

// Reduction float max value
void float_reduction(cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_input, size_t local_size);

// *****************************************************************************INTEGERS*****************************************************************************

// Integers kernel calls
void integer_kernel_calls(size_t input_size, cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, vector<integer> air_temperatures, size_t local_size);

// Reduction integers
void integer_reduction(cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_input, size_t local_size);


// ******************************************************************************************************************************************************************
// **************************************************************************MAIN EXECUTION**************************************************************************
// ******************************************************************************************************************************************************************


// Main execution
int main(int argc, char **argv)
{
#pragma region STARTUP - COMMAND LINE ARUGMENTS
	// Platform / device id
	int platform_id = 0;
	int device_id = 0;

	// Check the command line arguments
	for (int i = 1; i < argc; i++)
	{
		// Set the platform
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1)))
			platform_id = atoi(argv[++i]);

		// Set the device id
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1)))
			device_id = atoi(argv[++i]);

		// List the platform devices
		else if (strcmp(argv[i], "-l") == 0)
			cout << ListPlatformsDevices() << endl;

		// Print help to console
		else if (strcmp(argv[i], "-h") == 0)
			print_help();
	}
#pragma endregion

	// Detect any potential exceptions
	try
	{
		// Start of file reading
		hi_res_time_point start_of_execution = hi_res_clock::now();

		// Read in the data from the text file and parse
		vector<floating_point> air_temperatures = load_file_float(file);

		// Read in the data from the text file and parse
		vector<integer> air_temperatures_int = load_file_int(file);

		// Time taken to read and parse the file - converted to seconds
		auto time_elapsed_read_and_parse = chrono::duration_cast<chrono::milliseconds>(hi_res_clock::now() - start_of_execution).count() / milli_to_seconds;

		// Get the number of data entries
		number_of_data_entries = air_temperatures.size();

		// Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		cout << "***********************************************************************************************************************************************" << endl;
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id)					<< endl;
		cout << "***********************************************************************************************************************************************" << endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, "kernels.cl");
		cl::Program program(context, sources);

		// Build and debug the kernel code
		try
		{
			program.build();
		}

		// Catch any errors
		catch (const cl::Error& err)
		{
			cout << "Build Status: "	<< program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0])	<< endl;
			cout << "Build Options:\t"	<< program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0])	<< endl;
			cout << "Build Log:\t "		<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0])		<< endl;
			throw err;
		}

		// The following part adjusts the length of the input vector so it can be run for a specific workgroup size
		// If the total input length is divisible by the workgroup size
		// This makes the code more efficient
		size_t local_size = 128;
		size_t padding_size = air_temperatures.size() % local_size;

		// If the input vector is not a multiple of the local_size
		// Insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size)
		{
			// Create an extra vector with neutral values
			vector<float> air_temperatures_ext(local_size - padding_size, 0);
			vector<int> air_temperatures_ext_int(local_size - padding_size, 0);

			// Append that extra vector to our input
			air_temperatures.insert(air_temperatures.end(), air_temperatures_ext.begin(), air_temperatures_ext.end());
			air_temperatures_int.insert(air_temperatures_int.end(), air_temperatures_ext.begin(), air_temperatures_ext.end());
		}

		// Number of input elements
		size_t input_elements = air_temperatures.size();

		// Size in bytes
		size_t input_size_float = air_temperatures.size() * sizeof(floating_point);
		size_t input_size_int = air_temperatures_int.size() * sizeof(integer);

		// Number of groups
		size_t nr_groups = input_elements / local_size;

		// Device - input buffer
		cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, input_size_float);
		cl::Buffer buffer_input_int(context, CL_MEM_READ_ONLY, input_size_int);

		// Start fo float kernels
		hi_res_time_point start_of_float_execution = hi_res_clock::now();

		// Display 
		cout << "\n\nFLOAT KERNEL CALLS\n\n" << endl;

		// Execute the floating point kernels
		floating_point_kernel_calls(input_size_float, context, input_elements, queue, program, air_temperatures, local_size);

		// Time taken to execute float kernels - converted to seconds
		auto time_elapsed_float_kernels = chrono::duration_cast<chrono::milliseconds>(hi_res_clock::now() - start_of_float_execution).count() / milli_to_seconds;

		// Start fo float kernels
		hi_res_time_point start_of_int_execution = hi_res_clock::now();

		// Display 
		cout << "\n\nINTEGER KERNEL CALLS\n\n" << endl;

		// Execute the integer kernels
		integer_kernel_calls(input_size_int, context, input_elements, queue, program, air_temperatures_int, local_size);

		// Time taken to execute float kernels - converted to seconds
		auto time_elapsed_int_kernels = chrono::duration_cast<chrono::milliseconds>(hi_res_clock::now() - start_of_int_execution).count() / milli_to_seconds;
		
		// Time taken to execute kernels - converted to seconds
		auto time_elapsed_kernel = chrono::duration_cast<chrono::milliseconds>(hi_res_clock::now() - start_of_execution).count() / milli_to_seconds;

		// Device info
		device = context.getInfo<CL_CONTEXT_DEVICES>()[0];

		// Display time to read and parse the file
		cout << "***********************************************************************************************************************************************" << endl;
		cout << "Number of data entries: \t\t\t\t|| "				<< number_of_data_entries													<< endl;
		cout << "Preffered work group size: \t\t\t\t|| "			<< prefferSize																<< endl;
		cout << "Work group size:  \t\t\t\t\t|| "					<< local_size																<< endl;
		cout << "Time to read and parse the file:  \t\t\t|| "		<< time_elapsed_read_and_parse								<< " seconds"	<< endl;
		cout << "Time to execute float kernels:  \t\t\t|| "			<< time_elapsed_float_kernels								<< " seconds"	<< endl;
		cout << "Time to execute integer kernels:  \t\t\t|| "		<< time_elapsed_int_kernels									<< " seconds"	<< endl;
		cout << "Total time for all kernel executions:  \t\t\t|| "	<< time_elapsed_float_kernels + time_elapsed_int_kernels	<< " seconds"	<< endl;
		cout << "TOTAL PROGRAM EXECTUION TIME:  \t\t\t\t|| "		<< time_elapsed_kernel										<< " seconds"	<< endl;
		cout << "***********************************************************************************************************************************************" << endl;

		// Wait for key press to exit
		system("pause");
	}

	// Catch any errors
	catch (cl::Error err) { cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl; }

	// Success
	return 0;
}


// ******************************************************************************************************************************************************************
// ************************************************************************FUNCTION DEFINITIONS**********************************************************************
// ******************************************************************************************************************************************************************

// Print help
void print_help()
{
	cerr << "Application usage:" << endl;
	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

// Load file function
vector<floating_point> load_file_float(const char* file)
{
	// Vector of int for air temperature
	vector<floating_point> temperatures;

	// Input file stream
	ifstream ifs;

	// The read data
	string read_data;

	// Each line of the file
	string line;

	// If the directory exsists - open it
	if (file != nullptr)
		ifs.open(file);

	// If the stream is open
	// get each line and add it to the data string
	if (ifs.is_open())
		while (getline(ifs, line))
			temperatures.push_back(parse_string_to_float(line));

	// Close the stream and return the data
	ifs.close();
	return temperatures;
}

// Parse each line of the file
floating_point parse_string_to_float(string line)
{
	// Air temperature
	float temperature;

	// Delimiter
	char delimiter = ' ';

	// Counter for the number of spaces in the line of text
	int delimiter_count = 0;

	// String to hold the temperatures
	string number_to_string;

	// Loop through the data in the string
	for (int i = 0; i < line.length(); i++)
	{
		// Less than 5 spaces - not reached the temperature data
		if (delimiter_count < 5)
		{
			// If the data is a whitespace - increase spaces count
			if (line[i] == delimiter) delimiter_count++;
		}

		// Reached the temperature data
		else
		{
			// Else read in the temperature data
			number_to_string += line[i];
		}
	}

	// Convert and push the data to the vector
	temperature = stof(number_to_string);

	// Return the vector of air temperatures
	return temperature;
}

// Load file function
vector<integer> load_file_int(const char* file)
{
	// Vector of int for air temperature
	vector<integer> temperatures;

	// Input file stream
	ifstream ifs;

	// The read data
	string read_data;

	// Each line of the file
	string line;

	// If the directory exsists - open it
	if (file != nullptr)
		ifs.open(file);

	// If the stream is open
	// get each line and add it to the data string
	if (ifs.is_open())
		while (getline(ifs, line))
			temperatures.push_back(parse_string_to_int(line));

	// Close the stream and return the data
	ifs.close();
	return temperatures;
}

// Parse each line of the file
integer parse_string_to_int(string line)
{
	// Air temperature
	int temperature;

	// Delimiter
	char delimiter = ' ';

	// Counter for the number of spaces in the line of text
	int delimiter_count = 0;

	// String to hold the temperatures
	string number_to_string;

	// Loop through the data in the string
	for (int i = 0; i < line.length(); i++)
	{
		// Less than 5 spaces - not reached the temperature data
		if (delimiter_count < 5)
		{
			// If the data is a whitespace - increase spaces count
			if (line[i] == delimiter) delimiter_count++;
		}

		// Reached the temperature data
		else
		{
			// Else read in the temperature data
			number_to_string += line[i];
		}
	}

	// Convert and push the data to the vector
	temperature = (int)(stof(number_to_string) * 10);

	// Return the vector of air temperatures
	return temperature;
}

// *******************************************************************************FLOATS*****************************************************************************

// Floating point kernel calls
void floating_point_kernel_calls(size_t input_size, cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, vector<floating_point> air_temperatures, size_t local_size)
{
	// Device - input buffer
	cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, input_size);

	// Copy temperatures arrays to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, input_size, &air_temperatures[0]);

	// Reduction max kernel call
	float_reduction(context, input_elements, queue, program, buffer_input, local_size);
}

// Reduction floats
void float_reduction(cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_input, size_t local_size)
{
#pragma region REDUCTION MAX FLOATS
	// Host - output
	vector<floating_point> temperature_redux_max_result(input_elements);

	// Size in bytes
	size_t output_size = temperature_redux_max_result.size() * sizeof(floating_point);

	// Device - output buffers
	cl::Buffer buffer_output_redux_max(context, CL_MEM_READ_WRITE, output_size);

	// Zero output buffers on device memory
	queue.enqueueFillBuffer(buffer_output_redux_max, 0, 0, output_size);

	// Assign an ulong for holding the execution time of kernels and int for kernel calls
	cl_ulong execution_time;
	cl_ulong transfer_time;
	cl_ulong total_execution_time;
	int kernel_launches = 1;
	bool reduced = false;

	// Dsiaply info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "MAX REDUCTION FLOATS" << endl;

	// Kernel intialisation
	cl::Kernel kernel_redux_max = cl::Kernel(program, "reduction_max");
	kernel_redux_max.setArg(0, buffer_input);
	kernel_redux_max.setArg(1, buffer_output_redux_max);
	kernel_redux_max.setArg(2, cl::Local(local_size * sizeof(floating_point)));

	// Call all kernels in a sequence
	cl::Event event_redux_max_profiling;
	cl::Event event_redux_max_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_max_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_max, CL_TRUE, 0, output_size, &temperature_redux_max_result[0], NULL, &event_redux_max_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_max_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_max_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_max_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_max_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	total_execution_time = execution_time;
	cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;

	// loop through workgroups and reduce
	while(!reduced)
	{
		// If the reduction is complete
		if (temperature_redux_max_result[1] == 0.0f) break;/*reduced = true;*/

		kernel_redux_max.setArg(0, buffer_output_redux_max);
		kernel_redux_max.setArg(1, buffer_output_redux_max);
		kernel_redux_max.setArg(2, cl::Local(local_size * sizeof(floating_point)));

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_redux_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_max_profiling);

		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_output_redux_max, CL_TRUE, 0, output_size, &temperature_redux_max_result[0], NULL, &event_redux_max_transfer);

		// Display the profiling event data for the kernel
		execution_time = event_redux_max_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_max_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		transfer_time = event_redux_max_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_max_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total_execution_time += execution_time;
		kernel_launches++;
		cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;
	}

	// Display the profiling event data for the kernel
	cout << "Total reduction kernel luanches: " << kernel_launches << "\t|| Total time for " << kernel_launches << " executions [nano-seconds]: "	<< total_execution_time				<< endl;
	cout << "MAX TEMPERATURE: "																														<< temperature_redux_max_result[0]	<< endl;
	cout << "***********************************************************************************************************************************************"							<< endl;
#pragma endregion

#pragma region REDUCTION MIN FLOATS
	// Host - output
	vector<floating_point> temperature_redux_min_result(input_elements);

	// Device - output buffers
	cl::Buffer buffer_output_redux_min(context, CL_MEM_READ_WRITE, output_size);

	// Assign an ulong for holding the execution time of kernels and int for kernel calls
	execution_time = 0;
	transfer_time = 0;
	total_execution_time = 0;
	kernel_launches = 1;
	reduced = false;

	// Dsiaply info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "MIN REDUCTION FLOATS" << endl;

	// Kernel intialisation
	cl::Kernel kernel_redux_min = cl::Kernel(program, "reduction_min");
	kernel_redux_min.setArg(0, buffer_input);
	kernel_redux_min.setArg(1, buffer_output_redux_min);
	kernel_redux_min.setArg(2, cl::Local(local_size * sizeof(floating_point)));

	// Call all kernels in a sequence
	cl::Event event_redux_min_profiling;
	cl::Event event_redux_min_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_min_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_min, CL_TRUE, 0, output_size, &temperature_redux_min_result[0], NULL, &event_redux_min_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_min_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_min_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_min_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_min_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	total_execution_time = execution_time;
	cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;

	// While the element at the local size position is not 0 reduce the min
	while (!reduced)
	{
		// If the reduction is complete
		if (temperature_redux_min_result[1] == 0.0f) break; /*reduced = true;*/

		kernel_redux_min.setArg(0, buffer_output_redux_min);
		kernel_redux_min.setArg(1, buffer_output_redux_min);
		kernel_redux_min.setArg(2, cl::Local(local_size * sizeof(floating_point)));

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_redux_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_min_profiling);

		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_output_redux_min, CL_TRUE, 0, output_size, &temperature_redux_min_result[0], NULL, &event_redux_min_transfer);

		// Display the profiling event data for the kernel
		execution_time = event_redux_min_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_min_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();	
		transfer_time = event_redux_min_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_min_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total_execution_time += execution_time;
		kernel_launches++;
		cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;
	}

	// Display the profiling event data for the kernel
	cout << "Total reduction kernel luanches: " << kernel_launches << "\t|| Total time for " << kernel_launches << " executions [nano-seconds]: "	<< total_execution_time				<< endl;
	cout << "MIN TEMPERATURE: "																														<< temperature_redux_min_result[0]	<< endl;
	cout << "***********************************************************************************************************************************************"							<< endl;
#pragma endregion

#pragma region REDUCTION SUM FLOATS
	// Host - output
	vector<floating_point> temperature_redux_sum_result(input_elements);

	// Device - output buffers
	cl::Buffer buffer_output_redux_sum(context, CL_MEM_READ_WRITE, output_size);

	// Assign an ulong for holding the execution time of kernels and int for kernel calls
	execution_time = 0;
	transfer_time = 0;
	total_execution_time = 0;
	kernel_launches = 1;
	reduced = false;

	// Dsiaply info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "MEAN REDUCTION FLOATS" << endl;

	// Kernel intialisation
	cl::Kernel kernel_redux_sum = cl::Kernel(program, "reduction_sum");
	kernel_redux_sum.setArg(0, buffer_input);
	kernel_redux_sum.setArg(1, buffer_output_redux_sum);
	kernel_redux_sum.setArg(2, cl::Local(local_size * sizeof(floating_point)));

	// Call all kernels in a sequence
	cl::Event event_redux_sum_profiling;
	cl::Event event_redux_sum_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_sum_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_sum, CL_TRUE, 0, output_size, &temperature_redux_sum_result[0], NULL, &event_redux_sum_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_sum_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_sum_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_sum_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_sum_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	total_execution_time = execution_time;
	cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;

	// While the element at the local size position is not 0 reduce the min
	while (!reduced)
	{
		// If the reduction is complete
		if (temperature_redux_sum_result[1] == 0.0f) break;/*reduced = true;*/

		kernel_redux_sum.setArg(0, buffer_output_redux_sum);
		kernel_redux_sum.setArg(1, buffer_output_redux_sum);
		kernel_redux_sum.setArg(2, cl::Local(local_size * sizeof(floating_point)));

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_redux_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_sum_profiling);

		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_output_redux_sum, CL_TRUE, 0, output_size, &temperature_redux_sum_result[0], NULL, &event_redux_sum_transfer);

		// Display the profiling event data for the kernel
		execution_time = event_redux_sum_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_sum_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		transfer_time = event_redux_sum_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_sum_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total_execution_time += execution_time;
		kernel_launches++;
		cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;
	}

	// Calculate means
	mean_float = temperature_redux_sum_result[0] / number_of_data_entries;
	cout << "Total reduction kernel luanches: " << kernel_launches << "\t|| Total time for " << kernel_launches << " executions [nano-seconds]: "	<< total_execution_time << endl;
	cout << "MEAN TEMPERATURE: "																													<< mean_float			<< endl;
	cout << "***********************************************************************************************************************************************"				<< endl;
#pragma endregion

#pragma region REDUCTION STANDARD DEVIATION FLOATS
	// Host - output
	vector<floating_point> temperature_redux_std_dev_result(input_elements);

	// Device - output buffers
	cl::Buffer buffer_output_redux_std_dev(context, CL_MEM_READ_WRITE, output_size);

	// Assign an ulong for holding the execution time of kernels and int for kernel calls
	execution_time = 0;
	transfer_time = 0;
	total_execution_time = 0;
	kernel_launches = 1;
	reduced = false;

	// Dsiaply info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "STANDARD DEVIATION REDUCTION FLOATS" << endl;

	// Kernel initialisation
	cl::Kernel kernel_redux_std_dev = cl::Kernel(program, "reduction_standard_deviation");
	kernel_redux_std_dev.setArg(0, buffer_input);
	kernel_redux_std_dev.setArg(1, buffer_output_redux_std_dev);
	kernel_redux_std_dev.setArg(2, cl::Local(local_size * sizeof(floating_point)));
	kernel_redux_std_dev.setArg(3, mean_float);

	// Call all kernels in a sequence
	cl::Event event_redux_std_dev_profiling;
	cl::Event event_redux_std_dev_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_std_dev, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_std_dev_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_std_dev, CL_TRUE, 0, output_size, &temperature_redux_std_dev_result[0], NULL, &event_redux_std_dev_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_std_dev_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_std_dev_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_std_dev_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_std_dev_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	total_execution_time = execution_time;
	cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;

	// While the element at the local size position is not 0 reduce the sum
	while (!reduced)
	{
		// If the reduction is complete
		if (temperature_redux_std_dev_result[1] == 0.0f) break; /*reduced = true;*/

		kernel_redux_sum.setArg(0, buffer_output_redux_std_dev);
		kernel_redux_sum.setArg(1, buffer_output_redux_std_dev);
		kernel_redux_sum.setArg(2, cl::Local(local_size * sizeof(floating_point)));

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_redux_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_std_dev_profiling);

		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_output_redux_std_dev, CL_TRUE, 0, output_size, &temperature_redux_std_dev_result[0], NULL, &event_redux_std_dev_transfer);

		// Display the profiling event data for the kernel
		execution_time = event_redux_std_dev_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_std_dev_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		transfer_time = event_redux_std_dev_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_std_dev_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		total_execution_time += execution_time;
		kernel_launches++;
		cout << "Kernel luanch: " << kernel_launches << "\t\t\t|| Time for kernel " << kernel_launches << " execution [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;
	}

	// Calculate means
	variance_float = temperature_redux_std_dev_result[0] / number_of_data_entries;
	cout << "Total reduction kernel luanches: " << kernel_launches << "\t|| Total time for " << kernel_launches << " executions [nano-seconds]: "	<< total_execution_time << endl;
	cout << "VARIANCE: "																															<< variance_float		<< endl;
	cout << "STANDARD DEVIATION: "																													<< sqrt(variance_float) << endl;
	cout << "***********************************************************************************************************************************************"				<< endl;
#pragma endregion
}

// *****************************************************************************INTEGERS*****************************************************************************

// Integer kernel calls
void integer_kernel_calls(size_t input_size, cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, vector<integer> air_temperatures, size_t local_size)
{
	// Device - input buffer
	cl::Buffer buffer_input(context, CL_MEM_READ_WRITE, input_size);

	// Copy temperatures arrays to and initialise other arrays on device memory
	queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, input_size, &air_temperatures[0]);

	// Reduction kernel calls
	integer_reduction(context, input_elements, queue, program, buffer_input, local_size);
}

// Reduction integer value
void integer_reduction(cl::Context &context, size_t input_elements, cl::CommandQueue &queue, cl::Program &program, cl::Buffer &buffer_input, size_t local_size)
{
#pragma region REDUCTION MAX INTS
	// Host - output
	vector<integer> temperature_redux_max_result(input_elements);

	// Size in bytes
	size_t output_size = temperature_redux_max_result.size() * sizeof(integer);

	// Device - output buffers
	cl::Buffer buffer_output_redux_max(context, CL_MEM_READ_WRITE, output_size);

	// Zero output buffers on device memory
	queue.enqueueFillBuffer(buffer_output_redux_max, 0, 0, output_size);

	// Assign an ulong for holding the execution time of kernels
	cl_ulong transfer_time;
	cl_ulong execution_time;

	// Display info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "MAX REDUCTION INTEGERS - ATOMIC METHOD" << endl;

	// Kernel intialisation
	cl::Kernel kernel_redux_max = cl::Kernel(program, "reduction_max_int");
	kernel_redux_max.setArg(0, buffer_input);
	kernel_redux_max.setArg(1, buffer_output_redux_max);
	kernel_redux_max.setArg(2, cl::Local(local_size * sizeof(integer)));

	// Call all kernels in a sequence
	cl::Event event_redux_max_profiling;
	cl::Event event_redux_max_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_max_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_max, CL_TRUE, 0, output_size, &temperature_redux_max_result[0], NULL, &event_redux_max_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_max_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_max_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_max_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_max_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cout << "Total reduction kernel luanches: 1 \t|| Total time for all executions [nano-seconds]: "	<< execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time	<< endl;
	cout << "MAX TEMPERATURE: "																			<< (float)temperature_redux_max_result[0] / 10.0f								<< endl;
	cout << "***********************************************************************************************************************************************"							<< endl;

	// Preffered size
	prefferSize = kernel_redux_max.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
#pragma endregion

#pragma region REDUCTION MIN INTS
	// Host - output
	vector<integer> temperature_redux_min_result(input_elements);

	// Device - output buffers
	cl::Buffer buffer_output_redux_min(context, CL_MEM_READ_WRITE, output_size);

	// Display info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "MIN REDUCTION INTEGERS - ATOMIC METHOD" << endl;

	// Kernel intialisation
	cl::Kernel kernel_redux_min = cl::Kernel(program, "reduction_min_int");
	kernel_redux_min.setArg(0, buffer_input);
	kernel_redux_min.setArg(1, buffer_output_redux_min);
	kernel_redux_min.setArg(2, cl::Local(local_size * sizeof(integer)));

	// Call all kernels in a sequence
	cl::Event event_redux_min_profiling;
	cl::Event event_redux_min_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_min_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_min, CL_TRUE, 0, output_size, &temperature_redux_min_result[0], NULL, &event_redux_min_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_min_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_min_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_min_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_min_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cout << "Total reduction kernel luanches: 1 \t|| Total time for all executions [nano-seconds]: "	<< execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;
	cout << "MIN TEMPERATURE: "																			<< (float)temperature_redux_min_result[0] / 10.0f								<< endl;
	cout << "***********************************************************************************************************************************************"							<< endl;
#pragma endregion

#pragma region REDUCTION SUM INTS
	// Host - output
	vector<integer> temperature_redux_sum_result(input_elements);

	// Device - output buffers
	cl::Buffer buffer_output_redux_sum(context, CL_MEM_READ_WRITE, output_size);

	// Display info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "MEAN REDUCTION INTEGERS - ATOMIC METHOD" << endl;

	// Kernel intialisation
	cl::Kernel kernel_redux_sum = cl::Kernel(program, "reduction_sum_int");
	kernel_redux_sum.setArg(0, buffer_input);
	kernel_redux_sum.setArg(1, buffer_output_redux_sum);
	kernel_redux_sum.setArg(2, cl::Local(local_size * sizeof(integer)));

	// Call all kernels in a sequence
	cl::Event event_redux_sum_profiling;
	cl::Event event_redux_sum_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_sum_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_sum, CL_TRUE, 0, output_size, &temperature_redux_sum_result[0], NULL, &event_redux_sum_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_sum_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_sum_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_sum_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_sum_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	// Calculate means
	mean_float = (temperature_redux_sum_result[0] / 10.0f) / number_of_data_entries;
	mean_int = (int)(mean_float * 10.0f);
	cout << "Total reduction kernel luanches: 1 \t|| Total time for all executions [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time << endl;
	cout << "MEAN TEMPERATURE: " << mean_float << endl;
	cout << "***********************************************************************************************************************************************" << endl;
#pragma endregion

#pragma region REDUCTION STANDARD DEVIATION INTS
	// Host - output
	vector<integer> temperature_redux_std_dev_result(input_elements);

	// Device - output buffers
	cl::Buffer buffer_output_redux_std_dev(context, CL_MEM_READ_WRITE, output_size);

	// Display info
	cout << "***********************************************************************************************************************************************" << endl;
	cout << "STANDARD DEVIATION REDUCTION INTEGERS - ATOMIC METHOD" << endl;

	// Kernel intialisation
	cl::Kernel kernel_redux_std_dev = cl::Kernel(program, "reduction_standard_deviation_int");
	kernel_redux_std_dev.setArg(0, buffer_input);
	kernel_redux_std_dev.setArg(1, buffer_output_redux_std_dev);
	kernel_redux_std_dev.setArg(2, cl::Local(local_size * sizeof(integer)));
	kernel_redux_std_dev.setArg(3, mean_int);

	// Call all kernels in a sequence
	cl::Event event_redux_std_dev_profiling;
	cl::Event event_redux_std_dev_transfer;
	queue.enqueueNDRangeKernel(kernel_redux_std_dev, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &event_redux_std_dev_profiling);

	// Copy the result from device to host
	queue.enqueueReadBuffer(buffer_output_redux_std_dev, CL_TRUE, 0, output_size, &temperature_redux_std_dev_result[0], NULL, &event_redux_std_dev_transfer);

	// Display the profiling event data for the kernel
	execution_time = event_redux_std_dev_profiling.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_std_dev_profiling.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	transfer_time = event_redux_std_dev_transfer.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_redux_std_dev_transfer.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	// Calculate variance
	variance_float = (temperature_redux_std_dev_result[0] / 10.0f) / number_of_data_entries;
	cout << "Total reduction kernel luanches: 1 \t|| Total time for all executions [nano-seconds]: " << execution_time << "\t|| memory transfer [nano - seconds]: " << transfer_time			<< endl;
	cout << "VARIANCE: "																																			<< variance_float			<< endl;
	cout << "STANDARD DEVIATION: "																																	<< sqrt(variance_float)		<< endl;
	cout << "***********************************************************************************************************************************************"									<< endl;
#pragma endregion
}