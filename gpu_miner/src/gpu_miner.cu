#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

__global__ void findNonce(BYTE *difficulty, BYTE *block_content, size_t current_length, uint32_t *nonce) {
	// Calculating the thread_id
	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// Checking if the thread_id is within the range and the nonce is not already found
	if (thread_id > MAX_NONCE || *nonce != 0) {
		return;
	}

	char nonce_string[NONCE_SIZE];
	BYTE my_block_content[BLOCK_SIZE];
	BYTE block_hash[SHA256_HASH_SIZE];

	// Copying the block_content locally to avoid being overwritten by other threads
	d_strcpy((char*)my_block_content, (char*)block_content);

	// Computing the hash similar to the CPU code
	intToString(thread_id, nonce_string);
	d_strcpy((char*)my_block_content + current_length, nonce_string);
	apply_sha256(my_block_content, d_strlen((const char*)my_block_content), block_hash, 1);

	// Comparing the hash with the difficulty
	if (compare_hashes(block_hash, difficulty) <= 0) {
		*nonce = thread_id;
	}
}

int main(int argc, char **argv) {
	BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
			tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
			tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE];
	BYTE block_hash[SHA256_HASH_SIZE] = "0000000000000000000000000000000000000000000000000000000000000000"; // TODO: Update
	uint32_t nonce = 0; // TODO: Update
	size_t current_length;

	// Top hash
	apply_sha256(tx1, strlen((const char*)tx1), hashed_tx1, 1);
	apply_sha256(tx2, strlen((const char*)tx2), hashed_tx2, 1);
	apply_sha256(tx3, strlen((const char*)tx3), hashed_tx3, 1);
	apply_sha256(tx4, strlen((const char*)tx4), hashed_tx4, 1);
	strcpy((char *)tx12, (const char *)hashed_tx1);
	strcat((char *)tx12, (const char *)hashed_tx2);
	apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
	strcpy((char *)tx34, (const char *)hashed_tx3);
	strcat((char *)tx34, (const char *)hashed_tx4);
	apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
	strcpy((char *)tx1234, (const char *)hashed_tx12);
	strcat((char *)tx1234, (const char *)hashed_tx34);
	apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);

	// prev_block_hash + top_hash
	strcpy((char*)block_content, (const char*)prev_block_hash);
	strcat((char*)block_content, (const char*)top_hash);
	current_length = strlen((char*) block_content);

	cudaEvent_t start, stop;
	startTiming(&start, &stop);

	// Computing the required number of blocks for the given number of threads
	int threadsPerBlock = 512;
	int blocks = (int)(MAX_NONCE) / threadsPerBlock;

	if ((int)(MAX_NONCE) % threadsPerBlock != 0) {
		blocks++;
	}

	// Copying the block_content, difficulty to the device
	BYTE *device_block_content;
	cudaMalloc(&device_block_content, BLOCK_SIZE);
	cudaMemcpy(device_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);

	BYTE *difficulty;
	cudaMalloc(&difficulty, SHA256_HASH_SIZE);
	cudaMemcpy(difficulty, DIFFICULTY, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);

	// Creating a device variable to store the nonce
	uint32_t *device_nonce;
	cudaMalloc(&device_nonce, sizeof(uint32_t));
	cudaMemset(device_nonce, 0, sizeof(uint32_t));

	findNonce<<<blocks, threadsPerBlock>>>(difficulty, device_block_content, current_length, device_nonce);
	cudaDeviceSynchronize();

	// Copying the nonce from the device to the host
	cudaMemcpy(&nonce, device_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// Applying sha256 to the block_content with the found nonce
	char nonce_string[NONCE_SIZE];
	sprintf(nonce_string, "%u", nonce);
	strcpy((char*)block_content + current_length, nonce_string);
	apply_sha256(block_content, strlen((const char*)block_content), block_hash, 1);
	
	float seconds = stopTiming(&start, &stop);
	printResult(block_hash, nonce, seconds);

	cudaFree(device_block_content);
	cudaFree(difficulty);
	cudaFree(device_nonce);

	return 0;
}
