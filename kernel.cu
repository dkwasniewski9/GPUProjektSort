
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <chrono> 
#include <vector>
#include <iostream>
#include <iomanip> 
#include <algorithm>
#include <omp.h>


struct sortData {
    std::string sortName;
    float time;
};

void merge(int* array, int left, int mid, int right, int* temp) {
    int i, j, k;
    i = left;
    j = mid + 1;
    k = left;

    while (i <= mid && j <= right) {
        if (array[j] < array[i]) {
            temp[k] = array[j];
            ++j;
        }
        else {
            temp[k] = array[i];
            ++i;
        }
        ++k;
    }
    if (i <= mid) {
        while (i <= mid) {
            temp[k] = array[i];
            ++i;
            ++k;
        }
    }
    else {
        while (j <= right) {
            temp[k] = array[j];
            ++j;
            ++k;
        }
    }
    memcpy(array + left, temp + left, (right - left + 1) * sizeof(int));
}



void mergeSort(int* array, int left, int right, int* temp) {
    if (left < right) {
        int mid = (left + right) / 2;
        mergeSort(array, left, mid, temp);
        mergeSort(array, mid + 1, right, temp);
        merge(array, left, mid, right, temp);
    }
}

void mergeSortOMP(int* array, int left, int right, int* temp, int threads) {
    if (threads == 1)  mergeSort(array, left, right, temp);
    else if (threads > 1) {
        if (left < right) {
            int mid = (left + right) / 2;
            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSortOMP(array, left, mid, temp, threads / 2);
                #pragma omp section
                mergeSortOMP(array, mid + 1, right, temp, threads - threads / 2);
            }
            merge(array, left, mid, right, temp);
        }
    }
    
}

__device__ void mergeGPU(int* array, int left, int mid, int right, int* temp) {
    int i, j, k;
    i = left;
    j = mid + 1;
    k = left;

    while (i <= mid && j <= right) {
        if (array[j] < array[i]) {
            temp[k] = array[j];
            ++j;
        }
        else {
            temp[k] = array[i];
            ++i;
        }
        ++k;
    }
    if (i <= mid) {
        while (i <= mid) {
            temp[k] = array[i];
            ++i;
            ++k;
        }
    }
    else {
        while (j <= right) {
            temp[k] = array[j];
            ++j;
            ++k;
        }
    }
    memcpy(array + left, temp + left, (right - left + 1) * sizeof(int));
}

__device__ void mergeSort_GPU(int* array, int left, int right, int* temp)
{
    if (left < right) {
        int mid = (left + right) / 2;
        mergeSort_GPU(array, left, mid, temp);
        mergeSort_GPU(array, mid + 1, right, temp);
        __syncthreads();
        if(threadIdx.x == 0) mergeGPU(array, left, mid, right, temp);
        __syncthreads();
    }
}
__global__ void kernelMerge(int* array, int left, int right, int* temp) {
    mergeSort_GPU(array, left, right, temp);
}

cudaError_t mergeSortGPU(int* array, int left, int right) {
    int* gpuArray;
    int* gpuTempArray;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&gpuArray, (right + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&gpuTempArray, (right + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(gpuArray, array, (right + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    kernelMerge << <((right + 1) + 1024 - 1) / 1024, 1024 >> > (gpuArray, 0, right, gpuTempArray);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mergeSort launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(array, gpuArray, (right + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    if ((cudaStatus = cudaFree(gpuArray)) != cudaSuccess) {
        fprintf(stderr, "function: cudaFree, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
        exit(-1);
    }
    if ((cudaStatus = cudaFree(gpuTempArray)) != cudaSuccess) {
        fprintf(stderr, "function: cudaFree, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
        exit(-1);
    }


Error:
    cudaFree(gpuArray);
    cudaFree(gpuTempArray);

    return cudaStatus;
}


__global__ void rankSort(int* array, int* sortedArray, const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int x = 0;
    for (int j = 0; j < size; ++j) {
        if (array[i] > array[j]) ++x;
    }
    sortedArray[x] = array[i];
}

cudaError_t rankSortCuda(int* array, unsigned int size)
{
    int* gpuArray;
    int* gpuSortedArray;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpuArray, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&gpuSortedArray, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(gpuArray, array, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    rankSort << <(size + 1024 - 1) / 1024, 1024 >> > (gpuArray, gpuSortedArray, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rankSort launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(array, gpuSortedArray, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    if ((cudaStatus = cudaFree(gpuArray)) != cudaSuccess) {
        fprintf(stderr, "function: cudaFree, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
        exit(-1);
    }
    if ((cudaStatus = cudaFree(gpuSortedArray)) != cudaSuccess) {
        fprintf(stderr, "function: cudaFree, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
        exit(-1);
    }

Error:
    cudaFree(gpuArray);
    cudaFree(gpuSortedArray);

    return cudaStatus;
}


int* generateArray(int size) {
    int* array = new int[size];
    srand(11072022);
    for (int i = 0; i < size; ++i) {
        array[i] = i;
    }
    for (int i = 0; i < size - 1; i++)
    {
        int j = i + rand() / (RAND_MAX / (size - i) + 1);
        std::swap(array[i], array[j]);
    }
    return array;
}

void quicksort(int* array, int left, int right) {
    if (right <= left) return;

    int i = left - 1;
    int j = right + 1;
    int pivot = array[(left + right) / 2];

    while (true) {
        while (pivot > array[++i]);
        while (pivot < array[--j]);
        if (i <= j) std::swap(array[i], array[j]);
        else break;
    }
    if (j > left) quicksort(array, left, j);
    if (i < right) quicksort(array, i, right);
}


void ranksort(int* array, int size) {
    int x;
    int* sortedArray = new int[size];
    for (int i = 0; i < size; ++i) {
        x = 0;
        for (int j = 0; j < size; ++j) {
            if (array[i] > array[j]) ++x;
        }
        sortedArray[x] = array[i];
        //std::cout << "Finished :" << i << std::endl;
    }
    for (int i = 0; i < size; ++i) {
        array[i] = sortedArray[i];
    }
    delete[] sortedArray;
}

void ranksortOMP(int* array, int size) {
    int x;
    int i, j;
    int* sortedArray = new int[size];
    #pragma omp parallel for private(i, j, x) schedule(dynamic)
    for (i = 0; i < size; ++i) {
        //std::cout << i <<" "<< omp_get_thread_num() << std::endl;
        x = 0;
        for (j = 0; j < size; ++j) {
            if (array[i] > array[j]) ++x;
        }
        sortedArray[x] = array[i];
    }
    #pragma omp parallel for private(i) schedule(dynamic)
    for (i = 0; i < size; ++i) {
        array[i] = sortedArray[i];
    }
    delete[] sortedArray;
}

void oddEvenSort(int* array, int size) {
    int i, j;
    for (i = 0; i < size; ++i) {
        if (i % 2 == 1) {
            for (j = 1; j < size - 1; j += 2) {
                if (array[j] > array[j + 1]) {
                    std::swap(array[j], array[j + 1]);
                }
            }
        }
        else {
            for (j = 0; j < size - 1; j += 2) {
                if (array[j] > array[j + 1]) {
                    std::swap(array[j], array[j + 1]);
                }
            }
        }
    }
}

void oddEvenSortOMP(int* array, int size) {
    int i, j;
    for(i = 0; i < size; ++i){
        if (i % 2 == 1) {
            #pragma omp parallel for private(j) 
            for (j = 1; j < size - 1; j += 2) {
                if (array[j] > array[j + 1]) {
                    std::swap(array[j], array[j + 1]);
                }
            }
        }
        else {
            #pragma omp parallel for private(j)  
            for (j = 0; j < size - 1; j += 2) {
                if (array[j] > array[j + 1]) {
                    std::swap(array[j], array[j + 1]);
                }
            }
        }
    }

}

__global__ void oddEvenSort_GPU(int* array, int mode, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp;
    if (mode == 1) {
        if (array[index] > array[index + 1]) {
            tmp = array[index];
            array[index] = array[index + 1];
            array[index + 1] = tmp;
        }
    }
    else {
        if (array[index] > array[index + 1]) {
            tmp = array[index];
            array[index] = array[index + 1];
            array[index + 1] = tmp;
        }
    }
}

cudaError_t oddEvenSortGPU(int* array, int size) {
    int* gpuArray;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&gpuArray, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(gpuArray, array, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    for (int i = 0; i < size; ++i) {
        oddEvenSort_GPU << <(size + 1024 - 1) / 1024, 1024 >> > (gpuArray, i % 2, size);
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "oddEvenSort launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(array, gpuArray, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    if ((cudaStatus = cudaFree(gpuArray)) != cudaSuccess) {
        fprintf(stderr, "function: cudaFree, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
        exit(-1);
    }

Error:
    cudaFree(gpuArray);
    return cudaStatus;
}


int isSorted(int* array, int size) {
    for (int i = 0; i < size - 1; ++i) {
        if (array[i] > array[i + 1]) return 0;
    }
    return 1;
}


int main()
{
    int* array;
    int size = 100;
    int option;
    std::vector<sortData> times;
    thrust::device_vector<int> vector(size);
    sortData data;
    int* tempArray = new int[size];
    omp_set_num_threads(omp_get_max_threads()); 
    omp_set_nested(1);
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
    std::chrono::duration<float> time;
    std::cout << "Wybierz sortowanie:" << std::endl;
    std::cout << "1. Quicksort na CPU" << std::endl;
    std::cout << "2. Sortowanie z biblioteki standardowej" << std::endl;
    std::cout << "3. Rank sort CPU na 1 watku" << std::endl;
    std::cout << "4. Rank sort CPU dla maksymalnej liczby watkow: " << omp_get_max_threads() << std::endl;
    std::cout << "5. Rank sort GPU" << std::endl;
    std::cout << "6. Thrust Sort" << std::endl;
    std::cout << "7. Thrust Sort bez kopiowania danych" << std::endl;
    std::cout << "8. Merge sort CPU na 1 watku" << std::endl;
    std::cout << "9. Merge sort CPU dla maksymalnej liczby watkow: " << omp_get_max_threads() << std::endl;
    std::cout << "10. Merge sort GPU" << std::endl;
    std::cout << "11. Odd-even sort CPU na 1 watku" << std::endl;
    std::cout << "12. Odd-even sort CPU dla maksymalnej liczby watkow: " << omp_get_max_threads() << std::endl;
    std::cout << "13. Odd-even sort GPU" << std::endl;
    std::cout << "0. Zakoncz program" << std::endl;
    while (true) {
        std::cout << "Podaj numer sortowania: ";
        std::cin >> option;
        if (option == 0) break;
        array = generateArray(size);
        switch (option) {
        case 1:
            start = std::chrono::high_resolution_clock::now();
            quicksort(array, 0, size - 1);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Quicksort";
            data.time = time.count();
            times.push_back(data);
            break;
        case 2:
            start = std::chrono::high_resolution_clock::now();
            std::sort(array, array + size);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "std::sort";
            data.time = time.count();
            times.push_back(data);
            break;
        case 3:
            start = std::chrono::high_resolution_clock::now();
            ranksort(array, size);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Rank Sort CPU 1 watek";
            data.time = time.count();
            times.push_back(data);
            break;
        case 4:
            start = std::chrono::high_resolution_clock::now();
            ranksortOMP(array, size);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Rank Sort CPU max watkow";
            data.time = time.count();
            times.push_back(data);
            break;
        case 5:
            start = std::chrono::high_resolution_clock::now();
            rankSortCuda(array, size);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Rank Sort GPU";
            data.time = time.count();
            times.push_back(data);
            break;
        case 6:
            start = std::chrono::high_resolution_clock::now();
            thrust::copy(array, array + size, vector.begin());
            thrust::sort(vector.begin(), vector.end());
            thrust::copy(vector.begin(), vector.end(), array);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "thrust::sort GPU";
            data.time = time.count();
            times.push_back(data);
            break;
        case 7:
            thrust::copy(array, array + size, vector.begin());
            start = std::chrono::high_resolution_clock::now();
            thrust::sort(vector.begin(), vector.end());
            end = std::chrono::high_resolution_clock::now();
            thrust::copy(vector.begin(), vector.end(), array);
            time = end - start;
            data.sortName = "thrust::sort GPU";
            data.time = time.count();
            times.push_back(data);
            break;
        case 8:
            start = std::chrono::high_resolution_clock::now();
            mergeSort(array, 0, size - 1, tempArray);
            end = std::chrono::high_resolution_clock::now(); 
            time = end - start;
            data.sortName = "Merge Sort CPU 1 watek";
            data.time = time.count();
            times.push_back(data);
            break;
        case 9:
            start = std::chrono::high_resolution_clock::now();
            mergeSortOMP(array, 0, size - 1, tempArray, omp_get_max_threads());
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Merge Sort CPU max watkow";
            data.time = time.count();
            times.push_back(data);
            break;
        case 10:
            start = std::chrono::high_resolution_clock::now();
            mergeSortGPU(array, 0, size - 1);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Merge sort GPU";
            data.time = time.count();
            times.push_back(data);
            break;
        case 11:
            start = std::chrono::high_resolution_clock::now();
            oddEvenSort(array, size);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Odd-even Sort CPU 1 watek";
            data.time = time.count();
            times.push_back(data);
            break;
        case 12:
            start = std::chrono::high_resolution_clock::now();
            oddEvenSortOMP(array, size);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Odd-even Sort CPU max watkow";
            data.time = time.count();
            times.push_back(data);
            break;
        case 13:
            start = std::chrono::high_resolution_clock::now();
            oddEvenSortGPU(array, size);
            end = std::chrono::high_resolution_clock::now();
            time = end - start;
            data.sortName = "Odd-even Sort GPU";
            data.time = time.count();
            times.push_back(data);
            break;
        }
        if (isSorted(array, size) != 1) std::cout << "Nie udalo sie posortowac tablicy"<<std::endl;
        std::cout << times.back().sortName << ": ";
        std::cout << std::fixed << std::setprecision(8) << times.back().time << " s" << std::endl;
        delete[] array;
    }
    for (int i = 0; i < times.size(); ++i) {
        std::cout << times.at(i).sortName << ": ";
        std::cout << std::fixed << std::setprecision(8) << times.at(i).time<<" s"<<std::endl;
    }
    return 0;
}
