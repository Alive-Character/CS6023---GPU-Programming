#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/transform_scan.h>
#include <thrust/extrema.h>
#include <device_functions.h>

#define LLONG_MAX 1e9
using namespace std;
__global__ void print(int* a,int b,int c){
    for(int i = 0; i < b; i++){
        for(int j = 0; j < c;j++){
            printf("%d ",a[i*c+j]);
        }
        printf("\n");
    }

}
__global__ void printl(long long* a,int b,int c){
    for(int i = 0; i < b; i++){
        for(int j = 0; j < c;j++){
            printf("%d ",a[i*c+j]);
        }
        printf("\n");
    }

}

__global__ void nearest_city(long long* bellman_dist_flat, const int* flag, const int* lshell,   // maps shelter index to something
    const int* rcity, long long* travel, int* shelters_flat, int* cities_flat, int* locks, int num_cities, int num_shelters, int num_populated_cities)
{
    int city = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ long long min_dist;
    __shared__ int min_index;

    if (city >= num_cities || tid >= num_cities) return;

    if (tid == 0) {
        min_dist = LLONG_MAX;
        min_index = -1;
    }
    __syncthreads();

    if (flag[city] != 1 || flag[tid] != 2) return;
    long long dist = bellman_dist_flat[city * num_cities + tid];
    atomicMin(&min_dist, dist);
    __syncthreads();
    
    if (dist == min_dist) {
        atomicCAS(&min_index,-1, tid); // tie-breaking
    }
    __syncthreads();

    if (min_index == tid) {
        int s_idx = lshell[min_index];
        int c_idx = rcity[city];

        travel[c_idx] += bellman_dist_flat[city * num_cities + min_index];

        shelters_flat[s_idx * num_populated_cities + c_idx] = 1;
        cities_flat[c_idx * num_shelters + s_idx] = 1;
    }
}

__global__ void nearest_shelter(const long long* bellman_dist_flat, const int* flag, const int* lshell,   // maps shelter index to something
    const int* rcity, long long* travel, int* shelter_to_shelter, int* cities_flat, int num_cities, int num_shelters)
{
    int shelter = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ long long min_dist;
    __shared__ int min_index;
    if (shelter >= num_cities || tid >= num_cities) return;

    if (tid == 0) {
        min_dist = LLONG_MAX;
        min_index = -1;
    }
    __syncthreads();

    if (flag[shelter] != 2 || flag[tid] != 2) return;
    if(shelter == tid) return;
    long long dist = bellman_dist_flat[shelter * num_cities + tid];
    atomicMin(&min_dist, dist);
    __syncthreads();

    if (dist == min_dist) {
        atomicCAS(&min_index,-1, tid); // tie-breaking
    }
    __syncthreads();
    if (tid == min_index) {
        int s_idx = lshell[min_index];
        shelter_to_shelter[lshell[shelter]] = lshell[min_index];
    }
}


__global__ void shift_cities(const long long* bellman_dist, const int* lshell, long long* travel, 
    int* shelter_to_shelter, int* shelters, int* copy_shelters, int num_cities, int num_shelters, int num_populated_cities)
{
    int idx = blockIdx.x;
    if (idx >= num_shelters) return;
    int nearest_shelter = shelter_to_shelter[idx];
    for (int i = 0; i < num_populated_cities; i++)
    {
        if (shelters[idx * num_populated_cities + i] == 1)
        {
            copy_shelters[nearest_shelter * num_populated_cities + i] = 1;
            travel[i] += bellman_dist[lshell[idx] * num_cities + lshell[nearest_shelter]];
        }
    }
}

__global__ void alloc(int* shelter, long long* dist, int num_populated_cities, int max_distance_elderly, int num_shelter,
    long long* drops, long long* city, long long* pop, int* shelter_city, int* shelter_capacity, int* iterate)
{
    int shel_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (shel_num >= num_shelter || shelter_capacity[shel_num] == 0)
        return;
    for (int i = 0; i < num_populated_cities; i++)
    {
        int rcity = i;
        if (shelter[shel_num * num_populated_cities + i] == 0)  continue;
        if(pop[2* rcity ] + pop[2* rcity + 1] == 0) continue;
        iterate[rcity] += 1;
        int iter = iterate[rcity];
        if (dist[rcity] > max_distance_elderly)
        {
            drops[rcity * 24 + 0] = city[rcity];
            drops[rcity * 24 + 1] += 0;
            drops[rcity * 24 + 2] += pop[2 * rcity + 1];
            pop[2 * rcity + 1] = 0;
        }
        drops[rcity * 24 + iter * 3 + 0] = shelter_city[shel_num];
        if (shelter_capacity[shel_num] > pop[2 * rcity + 1])
        {
            drops[rcity * 24 + iter * 3 + 2] = pop[2 * rcity + 1];
            shelter_capacity[shel_num] -= pop[2 * rcity + 1];
            pop[2 * rcity + 1] = 0;
        }
        else
        {
            drops[rcity * 24 + iter * 3 + 2] = shelter_capacity[shel_num];
            pop[2 * rcity + 1] -= shelter_capacity[shel_num];
            shelter_capacity[shel_num] = 0;
        }
    }
    for (int i = 0; i < num_populated_cities; i++)
    {
        int rcity = i;
        if (shelter[shel_num * num_populated_cities + i] == 0)  continue;
        if(pop[2* rcity ] + pop[2* rcity + 1] == 0) continue;
        int iter = iterate[rcity];
        if (shelter_capacity[shel_num] > pop[2 * rcity])
        {
            drops[rcity * 24 + iter * 3 + 1] = pop[2 * rcity];
            shelter_capacity[shel_num] -= pop[2 * rcity];
            pop[2 * rcity] = 0;
            shelter[shel_num * num_populated_cities + i] = 0; //check
        }
        else
        {
            drops[rcity * 24 + iter * 3 + 1] = shelter_capacity[shel_num];
            pop[2 * rcity] -= shelter_capacity[shel_num];
            shelter_capacity[shel_num] = 0;
        }
    }
}

__global__ void check_if_full(int* shelter_capacity, long long* bellman_dist, int* shelter_city, int num_shelter, long long num_cities)
{
    if (blockDim.x * blockIdx.x + threadIdx.x >= num_shelter)
    {
        return;
    }
    if (shelter_capacity[blockDim.x * blockIdx.x + threadIdx.x] == 0)
    {
        bellman_dist[blockIdx.y * num_cities + shelter_city[blockDim.x * blockIdx.x + threadIdx.x]] = LLONG_MAX;
    }
}

__global__ void final_processing(long long* pop, long long* drops)
{
    int rcity = blockIdx.x;
    drops[rcity * 24 + 1] += pop[2 * rcity];
    drops[rcity * 24 + 2] += pop[2 * rcity + 1];
}

__global__ void init_drops(long long* drops, long long* city)
{
    int rcity = blockIdx.x;
    if (threadIdx.x == 0) {
        drops[rcity * 24 + threadIdx.x * 3 + 0] = city[rcity];
    }
    else {
        drops[rcity * 24 + threadIdx.x * 3 + 0] = -1;
    }
    drops[rcity * 24 + threadIdx.x * 3 + 1] = 0;
    drops[rcity * 24 + threadIdx.x * 3 + 2] = 0;
}

__global__ void check(long long* drops, long long* city, int num_populated_cities, long long* pop)
{
    int rcity = 29;
    if(rcity >= num_populated_cities)  return;
    drops[rcity * 24 + 0] = city[rcity];
    drops[rcity * 24 + 1] = pop[2 * rcity];
    drops[rcity * 24 + 2] = pop[2 * rcity + 1];
    pop[2 * rcity] = 0;
    pop[2 * rcity + 1] = 0;
}


__global__ void relax_edges(
    int* d_roads, int num_roads, long long* d_dist, int* d_prev,
    bool* d_changed, int* d_locks, int num_cities)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_roads) {
        int u = d_roads[tid * 4 + 0];
        int v = d_roads[tid * 4 + 1];
        int len = d_roads[tid * 4 + 2];
        //printf("%d %d \n",u,v);
        long long old_dist_u = d_dist[u];
        long long old_dist_v = d_dist[v];
        int old;

        // Relax u->v
        if (old_dist_u != LLONG_MAX && old_dist_u + len < old_dist_v) {
            do {
                old = atomicCAS(&d_locks[v], 0, 1);
                if (old == 0) {
                    if (old_dist_u + len < d_dist[v]) {
                        d_dist[v] = old_dist_u + len;
                        d_prev[v] = u;
                        *d_changed = true;
                        //printf("%d %d %d",v,u,old_dist_u + len);
                    }
                    d_locks[v] = 0;
                }
            } while (old != 0);
        }

        // Relax v->u (undirected)
        if (old_dist_v != LLONG_MAX && old_dist_v + len < old_dist_u) {
            do {
                old = atomicCAS(&d_locks[u], 0, 1);
                if (old == 0) {
                    if (old_dist_v + len < d_dist[u]) {
                        d_dist[u] = old_dist_v + len;
                        d_prev[u] = v;
                        *d_changed = true;
                    }
                    d_locks[u] = 0;
                }
            } while (old != 0);
        }
    }
}

__global__ void reconstruct_paths_kernel(
    const int* predecessors,  // [num_sources][num_nodes], flattened
    const long long* drops,       // [no_of_pop_cities][8][3]
    long long* paths,               // [no_of_pop_cities][7][max_path_length],
    long long* path_lengths,
    int num_cities,
    int max_path_length
) {
    long long i = blockIdx.x;
    long long j = threadIdx.x;

    if (drops[i * 24 + j * 3 + 3] == -1) {
        return; // add
    }
    long long src = drops[i * 24 + j * 3];
    long long dst = drops[i * 24 + j * 3 + 3];

    // Path storage (in reverse)
    long long* temp_path = new long long[num_cities];
    int length = 0;

    long long curr = dst;
    while (curr != -1 && length < max_path_length) {
        temp_path[length++] = curr;
        if (curr == src) break;
        curr = predecessors[src * num_cities + curr];
    }

    // If we reached the source, copy the path in correct order
    if (curr == src) {
        path_lengths[i * 7 + j] = length;
        for (int k = 0; k < length; ++k) {
            paths[i * 7 * num_cities + j * num_cities + k] = temp_path[length - 1 - k];
        }
    }
}

__global__ void path_l_kernel(long long* path_lengths, long long* total_lens,long long* num_drops) {
    long long i = blockIdx.x;

    long long sum = 1;
    for (int j = 0; j < 7; j++) {
        if (path_lengths[i * 7 + j] == 0) {
            num_drops[i] = j + 1;
            break;
        }
        sum += path_lengths[i * 7 + j]-1;  //check
       
    }
    total_lens[i] = sum;
}

__global__ void path_concatenate(long long* path_lengths, long long* path, long long* total_lens, long long* paths, long long num_cities) {
    long long i = blockIdx.x;
    int cnt = 1;
    paths[i * 7 * num_cities] = path[i * 7 * num_cities];
    for (long long j = 0; j < 7; j++) {
        for (long long k = 1; k < path_lengths[i * 7 + j]; k++) {
            paths[i * 7 * num_cities + cnt++] = path[i * 7 * num_cities + j * num_cities + k];
            //printf("%d %d\n",i,path[i * 7 * num_cities + j * num_cities + k]);
        }
    }
}

void gpu_shortest_paths(
    int* h_roads, int num_roads, int num_cities, int source,
    long long* h_dist, int* h_prev)
{
    int* d_roads;
    long long* d_dist;
    int* d_prev;
    bool* d_changed;
    int* d_locks;

    cudaMalloc(&d_roads, num_roads * 4 * sizeof(int));
    cudaMalloc(&d_dist, num_cities * sizeof(long long));
    cudaMalloc(&d_prev, num_cities * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));
    cudaMalloc(&d_locks, num_cities * sizeof(int));

    cudaMemcpy(d_roads, h_roads, num_roads * 4 * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize arrays
    long long* h_dist_init = new long long[num_cities];
    int* h_prev_init = new int[num_cities];
    int* h_locks_init = new int[num_cities];

    for (int i = 0; i < num_cities; i++) {
        h_dist_init[i] = LLONG_MAX;
        h_prev_init[i] = -1;
        h_locks_init[i] = 0;
    }
    h_dist_init[source] = 0;

    cudaMemcpy(d_dist, h_dist_init, num_cities * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev, h_prev_init, num_cities * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locks, h_locks_init, num_cities * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_roads + blockSize - 1) / blockSize;

    bool changed = true;
    int iter = 0;
    while (changed && iter < num_cities - 1) {
        changed = false;
        cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice);
        relax_edges << <gridSize, blockSize >> > (d_roads, num_roads, d_dist, d_prev, d_changed, d_locks, num_cities);
        cudaDeviceSynchronize();
        cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        iter++;
    }

    cudaMemcpy(h_dist, d_dist, num_cities * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prev, d_prev, num_cities * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_roads);
    cudaFree(d_dist);
    cudaFree(d_prev);
    cudaFree(d_changed);
    cudaFree(d_locks);
    delete[] h_dist_init;
    delete[] h_prev_init;
    delete[] h_locks_init;
}



int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]); // Read input file from command-line argument
    if (!infile) {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    long long num_cities;
    infile >> num_cities;

    long long num_roads;
    infile >> num_roads;

    int* flag = new int[num_cities];
    int* lshell = new int[num_cities];
    int* rcity = new int[num_cities];

    // Store roads as a flat array: [u1, v1, length1, capacity1, u2, v2, length2, capacity2, ...]
    int* roads = new int[num_roads * 4];

    for (int i = 0; i < num_roads; i++) {
        infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >> roads[4 * i + 3];
    }
    int* d_roads;
    cudaMalloc(&d_roads, sizeof(int) * num_roads * 4);
    cudaMemcpy(d_roads, roads, sizeof(int) * num_roads * 4, cudaMemcpyHostToDevice);


    int num_shelters;
    infile >> num_shelters;

    // Store shelters separately
    int* shelter_city = new int[num_shelters];
    int* shelter_capacity = new int[num_shelters];

    for (int i = 0; i < num_shelters; i++) {
        infile >> shelter_city[i] >> shelter_capacity[i];
        flag[shelter_city[i]] = 2;
        lshell[shelter_city[i]] = i;
    }

    int* d_shelter_city;
    cudaMalloc(&d_shelter_city, sizeof(int) * num_shelters);
    cudaMemcpy(d_shelter_city, shelter_city, sizeof(int) * num_shelters, cudaMemcpyHostToDevice);


    int* d_shelter_capacity;
    cudaMalloc(&d_shelter_capacity, sizeof(int) * num_shelters);
    cudaMemcpy(d_shelter_capacity, shelter_capacity, sizeof(int) * num_shelters, cudaMemcpyHostToDevice);

    int num_populated_cities;
    infile >> num_populated_cities;

    // Store populated cities separately
    long long* city = new long long[num_populated_cities];
    long long* pop = new long long[num_populated_cities * 2]; // Flattened [prime-age, elderly] pairs

    for (long long i = 0; i < num_populated_cities; i++) {
        infile >> city[i] >> pop[2 * i] >> pop[2 * i + 1];
        flag[city[i]] = 1;
        rcity[city[i]] = i;
    }
    long long* d_city;
    cudaMalloc(&d_city, sizeof(long long) * num_populated_cities);
    cudaMemcpy(d_city, city, sizeof(long long) * num_populated_cities, cudaMemcpyHostToDevice);

    long long* d_pop;
    cudaMalloc(&d_pop, sizeof(long long) * num_populated_cities*2);
    cudaMemcpy(d_pop, pop, sizeof(long long) * num_populated_cities*2, cudaMemcpyHostToDevice);


    int max_distance_elderly;
    infile >> max_distance_elderly;

    infile.close();

    long long* d_drops;
    cudaMalloc(&d_drops, sizeof(long long) * 24 * num_populated_cities);

    init_drops << <num_populated_cities, 8 >> > (d_drops, d_city);

    check << <1, 1 >> > (d_drops, d_city, num_populated_cities, d_pop);



    //Bellman
    long long* h_bellman_dist = new long long[num_cities * num_cities];
    int* h_predecessors = new int[num_cities * num_cities];
    for (int i = 0; i < num_cities; i++)
        gpu_shortest_paths(roads, num_roads, num_cities, i, h_bellman_dist + i * num_cities, h_predecessors + i * num_cities);


    long long* bellman_dist;
    int* predeccesors;

    cudaMalloc(&bellman_dist, sizeof(long long) * num_cities * num_cities);
    cudaMalloc(&predeccesors, sizeof(int) * num_cities * num_cities);

    cudaMemcpy(bellman_dist, h_bellman_dist, sizeof(long long) * num_cities * num_cities, cudaMemcpyHostToDevice);
    cudaMemcpy(predeccesors, h_predecessors, sizeof(int) * num_cities * num_cities, cudaMemcpyHostToDevice);


    // 2. flag
    int* d_flag;
    cudaMalloc(&d_flag, sizeof(int) * num_cities);
    cudaMemcpy(d_flag, flag, sizeof(int) * num_cities, cudaMemcpyHostToDevice);

    // 3. lshell
    int* d_lshell;
    cudaMalloc(&d_lshell, sizeof(int) * num_cities);
    cudaMemcpy(d_lshell, lshell, sizeof(int) * num_cities, cudaMemcpyHostToDevice);

    // 4. rcity
    int* d_rcity;
    cudaMalloc(&d_rcity, sizeof(int) * num_cities);
    cudaMemcpy(d_rcity, rcity, sizeof(int) * num_cities, cudaMemcpyHostToDevice);

    // 5. travel
    long long* d_travel;
    cudaMalloc(&d_travel, sizeof(long long) * num_populated_cities);

    // 6. shelters_flat
    int* d_shelters;
    cudaMalloc(&d_shelters, sizeof(int) * num_shelters * num_populated_cities);

    // 7. cities_flat
    int* d_cities;
    cudaMalloc(&d_cities, sizeof(int) * num_populated_cities * num_shelters);

    int* d_locks;
    cudaMalloc(&d_locks, num_cities * sizeof(int));
    cudaMemset(d_locks, 0, num_cities * sizeof(int)); // all locks are initially 0 (unlocked)


    nearest_city << < num_cities, num_cities >> > (bellman_dist, d_flag, d_lshell,
        d_rcity, d_travel, d_shelters, d_cities, d_locks, num_cities, num_shelters,num_populated_cities);
    
    int* d_iterate;
    cudaMalloc(&d_iterate, sizeof(int) * num_populated_cities);
    cudaMemset(d_iterate, 0, num_populated_cities * sizeof(int));

    alloc << < (num_shelters + 1023) / 1024, 1024 >> > (d_shelters, d_travel, num_populated_cities, max_distance_elderly, num_shelters,
        d_drops, d_city, d_pop, d_shelter_city, d_shelter_capacity, d_iterate);


    
    dim3 grid((num_shelters + 1023) / 1024, num_cities, 1);
    check_if_full << < grid, 1024 >> > (d_shelter_capacity, bellman_dist, d_shelter_city, num_shelters, num_cities);
    //cudaMemcpy(shelters, d_shelters, sizeof(int) * num_shelters * num_cities, cudaMemcpyDeviceToHost);

    int* d_copy_shelters;
    cudaMalloc(&d_copy_shelters, sizeof(int) * num_shelters * num_populated_cities);
    //cudaMemcpy(d_copy_shelter, d_shelters, sizeof(int) * num_shelters * num_cities, cudaMemcpyHostToDevice);

    int* shelter_to_shelter = new int[num_shelters];
    int* d_shelter_to_shelter;
    cudaMalloc(&d_shelter_to_shelter, num_shelters * sizeof(int));

    for (int i = 0; i < 6; i++)
    {
        nearest_shelter << <num_cities, num_cities >> > (bellman_dist, d_flag, d_lshell,
            d_rcity, d_travel, d_shelter_to_shelter, d_cities, num_cities, num_shelters);
            

        shift_cities << <num_shelters, 1 >> > (bellman_dist, d_lshell, d_travel,
            d_shelter_to_shelter, d_shelters, d_copy_shelters, num_cities, num_shelters, num_populated_cities);


        cudaMemcpy(d_shelters, d_copy_shelters, sizeof(int) * num_shelters * num_populated_cities, cudaMemcpyDeviceToDevice);
        alloc << < (num_shelters + 1023) / 1024, 1024 >> > (d_shelters, d_travel, num_populated_cities, max_distance_elderly, num_shelters,
            d_drops, d_city, d_pop, d_shelter_city, d_shelter_capacity, d_iterate);

        dim3 grid(1, num_cities, (num_shelters + 1023) / 1024);
        check_if_full << < grid, 1024 >> > (d_shelter_capacity, bellman_dist, d_shelter_city, num_shelters, num_cities);

    }

    final_processing << < num_populated_cities, 1 >> > (d_pop, d_drops);


    //path_reconstruuct
    long long* d_path;
    long long* d_path_lengths;
    cudaMalloc(&d_path, num_populated_cities * 7 * num_cities * sizeof(long long));
    cudaMalloc(&d_path_lengths, num_populated_cities * 7 * sizeof(long long));
    reconstruct_paths_kernel << <num_populated_cities, 7 >> > (predeccesors, d_drops, d_path, d_path_lengths, num_cities, num_cities);

    long long* total_lengths;
    long long* d_paths;
    long long* d_num_drops;


    cudaMalloc(&total_lengths, num_populated_cities * sizeof(long long));
    cudaMalloc(&d_num_drops, num_populated_cities * sizeof(long long));
    path_l_kernel << <num_populated_cities, 1 >> > (d_path_lengths, total_lengths, d_num_drops);

    cudaMalloc(&d_paths, num_populated_cities * 7*num_cities * sizeof(long long));
    path_concatenate << <num_populated_cities, 1 >> > (d_path_lengths, d_path, total_lengths,d_paths, num_cities);



    // set your answer to these variables
    long long* path_size = new long long[num_populated_cities];

    long long** paths = new long long* [num_populated_cities];
    for (int i = 0; i < num_populated_cities; ++i) {
        paths[i] = new long long[7 * num_cities];
    }
    long long* num_drops = new long long[num_populated_cities];

    long long*** drops = new long long** [num_populated_cities];
    for (int i = 0; i < num_populated_cities; ++i) {
        drops[i] = new long long* [8];
        for (int j = 0; j < 8; ++j) {
            drops[i][j] = new long long[3];
        }
    }

    //data_to_cpu
    long long* tempdrops = new long long[num_populated_cities * 8* 3];
    long long* temppaths = new long long[num_populated_cities * 7 * num_cities]; 
    cudaMemcpy(path_size, total_lengths, sizeof(long long) * num_populated_cities, cudaMemcpyDeviceToHost);
    cudaMemcpy(num_drops, d_num_drops, sizeof(long long) * num_populated_cities, cudaMemcpyDeviceToHost);
    cudaMemcpy(tempdrops, d_drops, sizeof(long long) * num_populated_cities * 8* 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(temppaths, d_paths, sizeof(long long) * num_populated_cities* 7 * num_cities, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_populated_cities; ++i) {
        for (int j = 0; j < 7 * num_cities; ++j) {
            paths[i][j] = temppaths[i * 7 * num_cities + j];
            //printf("%d ",temppaths[i * 7 * num_cities + j]);
        }
        //printf("\n %d \n",path_size[i]);
    }
    for (int i = 0; i < num_populated_cities; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) {
                drops[i][j][k] = tempdrops[i * 8 * 3 + j * 3 + k]; // Mapping the 1D index to 3D
                //printf("%d %d %d %d \n",i,j,k,tempdrops[i * 8 * 3 + j * 3 + k]);
            }
        }
    }




    cudaFree(bellman_dist);
    cudaFree(d_flag);
    cudaFree(d_lshell);
    cudaFree(d_rcity);
    cudaFree(d_travel);
    cudaFree(d_shelters);
    cudaFree(d_cities);



    ofstream outfile(argv[2]); // Read input file from command-line argument
    if (!outfile) {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }
    for (long long i = 0; i < num_populated_cities; i++) {
        long long currentPathSize = path_size[i];
        for (long long j = 0; j < currentPathSize; j++) {
            outfile << paths[i][j] << " ";
        }
        outfile << "\n";
    }

    for (long long i = 0; i < num_populated_cities; i++) {
        long long currentDropSize = num_drops[i];
        for (long long j = 0; j < currentDropSize; j++) {
            for (int k = 0; k < 3; k++) {
                outfile << drops[i][j][k] << " ";
            }
        }
        outfile << "\n";
    }




    return 0;
}
