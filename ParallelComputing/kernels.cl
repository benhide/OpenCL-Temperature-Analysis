// *************************************************************************************************************************************
// ************************************************************FLOATS*******************************************************************
// *************************************************************************************************************************************


// Reduction kernel to find the min value
kernel void reduction_max(global const float* input, global float* output, local float* local_aux)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// The group position relative to all other groups (globally)
	int group_id = get_group_id(0);

	// Cache all local values from global memory to local memory
	local_aux[local_id] = input[global_id];

	// Wait for all local threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)
	{
		// If the local id is less than the stride
		if (local_id < stride)
		{
			// If local value is less than the next local value - switch the values
			if (local_aux[local_id] < local_aux[local_id + stride])
			{
				local_aux[local_id] = local_aux[local_id + stride];
			}
		}

		// Wait for all local threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Assign local max to output at group index
	if (!local_id)
		output[group_id] = local_aux[local_id];

	//// Assign local max to output at group index
	//if (!local_id)
	//	if (local_aux[local_id] > output[0])
	//		output[0] = local_aux[local_id];
}

// Reduction kernel to find the max value 
kernel void reduction_min(global const float* input, global float* output, local float* local_aux)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// The group position relative to all other groups (globally)
	int group_id = get_group_id(0);

	// Cache all local values from global memory to local memory
	local_aux[local_id] = input[global_id];

	// Wait for all local threads to finish copying from global to local memory
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)
	{
		// If the local id is less than the stride
		if (local_id < stride)
		{
			// If local value is greater than the next local value - switch the values
			if (local_aux[local_id] > local_aux[local_id + stride])
			{
				local_aux[local_id] = local_aux[local_id + stride];
			}
		}

		// Wait for all local threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Assign local min to output at group index
	if (!local_id)
		output[group_id] = local_aux[local_id];

	//// Assign local max to output at group index
	//if (!local_id)
	//	if (local_aux[local_id] < output[0])
	//		output[0] = local_aux[local_id];
}

// Reduction kernel to find the sum value
kernel void reduction_sum(global const float* input, global float* output, local float* local_aux)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// The group position relative to all other groups (globally)
	int group_id = get_group_id(0);

	// Cache all local values from global memory to local memory
	local_aux[local_id] = input[global_id];

	// Wait for all local threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)
	{
		// If the local id is less than the stride - sum the values at local id and local id + the stride
		if (local_id < stride)
		{
			local_aux[local_id] += local_aux[local_id + stride];
		}

		// Wait for all local threads to finish copying from global to local memory
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Assign local sumn to output at group index
	if (!local_id)
		output[group_id] = local_aux[local_id];
}

// Reduction kernel to find the standard deviation sum value
kernel void reduction_standard_deviation(global const float* input, global float* output, local float* local_aux , float mean)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// The group position relative to all other groups (globally)
	int group_id = get_group_id(0);

	// Calculate the sqaure of values minus the mean - Cache all local values from global memory to local memory
	local_aux[local_id] = (input[global_id] - mean) * (input[global_id] - mean);

	// Wait for all local threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)
	{
		// If the local id is less than the stride - sum the values at local id and local id + the stride
		if (local_id < stride)
		{
			local_aux[local_id] += local_aux[local_id + stride];
		}

		// Wait for all local threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Assign local sumn to output at group index
	if (!local_id)
		output[group_id] = local_aux[local_id];
}


// *************************************************************************************************************************************
// ************************************************************INTEGERS*****************************************************************
// *************************************************************************************************************************************


// Reduction kernel to find the min value
kernel void reduction_max_int(global const int* input, global int* output, local int* local_aux)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// Cache all local values from global memory to local memory
	local_aux[local_id] = input[global_id];

	// Wait for all local threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)
	{
		// If the local id is less than the stride
		if (local_id < stride)
		{
			// If local value is less than the next local value - switch the values
			if (local_aux[local_id] < local_aux[local_id + stride])
			{
				local_aux[local_id] = local_aux[local_id + stride];
			}
		}

		// Wait for all local threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Local value or global value highest - atomic method
	if (!local_id)
		atomic_max(&output[0], local_aux[local_id]);
}

// Reduction kernel to find the max value 
kernel void reduction_min_int(global const int* input, global int* output, local int* local_aux)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// Cache all local values from global memory to local memory
	local_aux[local_id] = input[global_id];

	// Wait for all local threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)
	{
		// If the local id is less than the stride
		if (local_id < stride)
		{
			// If local value is greater than the next local value - switch the values
			if (local_aux[local_id] > local_aux[local_id + stride])
			{
				local_aux[local_id] = local_aux[local_id + stride];
			}
		}

		// Wait for all local threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Local value or global value lowest - atomic method
	if (!local_id)
		atomic_min(&output[0], local_aux[local_id]);
}

// Reduction kernel to find the sum value
kernel void reduction_sum_int(global const int* input, global int* output, local int* local_aux)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// Cache all local values from global memory to local memory
	local_aux[local_id] = input[global_id];

	// Wait for all local threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)

	{	// If the local id is less than the stride - sum the values at local id and local id + the stride
		if (local_id < stride)
		{
			local_aux[local_id] += local_aux[local_id + stride];
		}

		// Wait for all local threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Sum the values - atomic method
	if (!local_id)
		atomic_add(&output[0], local_aux[local_id]);
}

// Reduction kernel to find the standard deviation sum value
kernel void reduction_standard_deviation_int(global const int* input, global int* output, local int* local_aux, int mean)
{
	// Current thread
	int global_id = get_global_id(0);

	// Local work item ID
	int local_id = get_local_id(0);

	// Local work-items count
	int local_size = get_local_size(0);

	// Calculate the sqaure of values minus the mean - Cache all local values from global memory to local memory
	// Divide by 10 due to multiplication of ints (multiplication to avoid loss of precision)
	local_aux[local_id] = (input[global_id] - mean) * (input[global_id] - mean) / 10;

	// Wait for all local threads to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through local memory - coalesced memory access
	for (int stride = local_size / 2; stride > 0; stride /= 2)
	{
		// If the local id is less than the stride - sum the values at local id and local id + the stride
		if (local_id < stride)
		{
			local_aux[local_id] += local_aux[local_id + stride];
		}
	
		// Wait for all local threads to finish
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Sum the values - atomic method
	if (!local_id)
		atomic_add(&output[0], local_aux[local_id]);
}


// *************************************************************************************************************************************
// ************************************************************SORTING******************************************************************
// *************************************************************************************************************************************


//// Sorting kernel to sort the input data - integers - selection sort using global memory
//// Reference - http://www.bealto.com/gpu-sorting_parallel-selection.html - Eric Bainville - June 2011
//kernel void parallel_selection_sort(global const float* input, global float* output, local float* local_aux)
//{
//	// Current thread
//	int global_id = get_global_id(0);
//
//	// Global size
//	int global_size = get_global_size(0);
//
//	// Input size
//	int local_size = get_local_size(0);
//
//	// Block size
//	int block_size = local_size;
//
//	// Store local value
//	float i_key = input[global_id];
//
//	// Compute position of output
//	int position = 0;
//
//	// Loop through the data
//	for (int j = 0; j < global_size; j += block_size)
//	{
//		// Ensures all threads have finished the processing loop before loading a new block
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		//
//		for (int i = get_local_id(0); i < block_size; i += local_size)
//			local_aux[i] = input[j + i];
//
//		// Ensures the block is entirely loaded before starting the next processing loop
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		//
//		for (int i = 0; i < block_size; i++)
//		{
//			// Key for j
//			float j_key = local_aux[i];
//
//			// Is j key smaller than i key?
//			bool j_key_is_smaller = (j_key < i_key) || (j_key == i_key && (j + i) < global_id);
//
//			// Set position to 1 or 0 depending on the result of the smaller bool operation
//			if (j_key_is_smaller) position += 1;
//			else position += 0;
//		}
//	}
//
//	// Set the output at the position as the i key
//	output[position] = i_key;
//}
//
//// Compare and exchange
//void cmpxchg(global int* input_one, global int* input_two, bool direction)
//{
//	// Swap values - depend on ascending/descending and highest value
//	if ((!direction && *input_one > *input_two) || (direction && *input_one < *input_two))
//	{
//		int temp = *input_one;
//		*input_one = *input_two;
//		*input_two = temp;
//
//		// Wait for all local threads to finish
//		barrier(CLK_GLOBAL_MEM_FENCE);
//	}
//}
//
//// Merge the arrays
//void bitonic_merge(int global_id, global int* input, int number, bool direction)
//{
//	// Loop through ...
//	for (int i = number / 2; i > 0; i /= 2)
//	{
//		// Compare and exchange
//		if ((global_id % (i * 2)) < i)
//			cmpxchg(&input[global_id], &input[global_id + i], direction);
//
//		// Wait for all local threads to finish
//		barrier(CLK_GLOBAL_MEM_FENCE);
//	}
//}
//
//// Bitonic sort
//kernel void sort_bitonic(global int* A) 
//{
//	int id = get_global_id(0);
//	int N = get_global_size(0);
//
//	for (int i = 1; i < N / 2; i *= 2) 
//	{
//		if (id % (i * 4) < i * 2)
//			bitonic_merge(id, A, i * 2, false);
//
//		else if ((id + i * 2) % (i * 4) < i * 2)
//			bitonic_merge(id, A, i * 2, true);
//
//		barrier(CLK_GLOBAL_MEM_FENCE);
//	}
//
//	if (id == 0)
//		bitonic_merge(id, A, N, false);
//}
//
//// Odd even sort - compare exchange
//void cmpxchg_oddeven(global int* A, global int* B)
//{
//	if (*A > *B) 
//	{
//		int t = *A; 
//		*A = *B; 
//		*B = t;
//	}
//}
//
//// Odd even sort
//kernel void sort_oddeven(global int* A)
//{
//	int id = get_global_id(0); int N = get_global_size(0);
//	for (int i = 0; i < N; i += 2)
//	{
//		// Odd
//		if (id % 2 == 0 && id + 1 < N)
//			cmpxchg_oddeven(&A[id], &A[id + 1]);
//
//		// Even
//		if (id % 2 == 1 && id + 1 < N) 
//			cmpxchg_oddeven(&A[id], &A[id + 1]);
//	}
//}