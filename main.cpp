#include <iostream>
#include <cassert>
#include <iomanip>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <align.h>

using namespace std;

// I stole this number from the ATLAS assembly.
const int PREFETCH_DISTANCE = 80;
//const int PREFETCH_DISTANCE = 32;

const int NUM_TRIALS = 500;

   template<typename DotFunction>
   float time_dot(DotFunction f, int N, float *A, float *B) {
   auto total_sec = 0;
   auto total_usec = 0;

   for(int i = 0; i < NUM_TRIALS; ++i) {
       struct timeval tv, _tv;
       gettimeofday(&_tv,NULL);
       auto dot = f(N, A, B);
       gettimeofday(&tv,NULL);
       // A relatively harmless call to keep the compiler from realizing
       // we don't actually do any useful work with the dot product.
       srand48(dot);
       total_sec +=(tv.tv_sec - _tv.tv_sec);	
       total_usec += (tv.tv_usec - _tv.tv_usec);
   }
       auto total =  total_sec * 1000000 + total_usec; 

   return float(total) / (1e6 * NUM_TRIALS);
   }

// Generate a random vector
float *generate_vector(int N) {
    float *v = (float *)malloc_aligned(32,
	    sizeof(float) * N);

    for(int i = 0; i < N; ++i) {
	v[i] = drand48();
    }

    return v;
}

float simple_dot(int, float *, float *);
float simple_prefetch_dot(int, float *, float *);
float unroll_dot(int, float *, float *);
float sse_dot(int, float *, float *);
float avx_dot(int, float *, float *);
float avx_unroll_dot(int, float *, float *);
float avx_unroll_prefetch_dot(int, float *, float *);
float blas_dot(int, float *, float *);
float cublas_dot(int, float *, float *);

int main() {
    auto N = (128 << 20) / sizeof(float);

    cout << "Generating " << N << " element vectors." << endl;

    auto A = generate_vector(N);
    auto B = generate_vector(N);

    assert(A);
    assert(B);

#define TIME(f) cout << setw(24) << #f "\t" << time_dot(f, N, A, B) << endl;

    TIME(simple_dot);
    TIME(simple_prefetch_dot);
    TIME(unroll_dot);
    TIME(sse_dot);
    TIME(avx_dot);
    TIME(avx_unroll_dot);
    TIME(avx_unroll_prefetch_dot);
    cout<<"dude!!"<<endl;
    free_aligned(A);
    free_aligned(B);
}

// Returns the time in seconds to compute the dot product.
//
// This is the usual implementation.
float simple_dot(int N, float *A, float *B) {
    float dot = 0;
    for(int i = 0; i < N; ++i) {
	dot += A[i] * B[i];
    }

    return dot;
}

float unroll_dot(int N, float *A, float *B) {
    float dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0;

    for(int i = 0; i < N; i += 4) {
	dot1 += A[i] * B[i];
	dot2 += A[i + 1] * B[i + 1];
	dot3 += A[i + 2] * B[i + 2];
	dot4 += A[i + 3] * B[i + 3];
    }

    return dot1 + dot2 + dot3 + dot4;
}

float simple_prefetch_dot(int N, float *A, float *B) {
    float dot = 0;
    for(int i = 0; i < N; ++i) {
	__builtin_prefetch(A + i + PREFETCH_DISTANCE, 0, 0);
	dot += A[i] * B[i];
    }

    return dot;
}

float sse_dot(int N, float *A, float *B) {
    const int VECTOR_SIZE = 4;

    typedef float vec
	__attribute__ ((vector_size (sizeof(float) * VECTOR_SIZE)));

    vec temp = {0};

    N /= VECTOR_SIZE;

    vec *Av = (vec *)A;
    vec *Bv = (vec *)B;

    for(int i = 0; i < N; ++i) {
	temp += *Av * *Bv;

	Av++;
	Bv++;
    }

    union {
	vec tempv;
	float tempf[VECTOR_SIZE];
    };

    tempv = temp;

    float dot = 0;
    for(int i = 0; i < VECTOR_SIZE; ++i) {
	dot += tempf[i];
    }

    return dot;
}

float avx_dot(int N, float *A, float *B) {
    const int VECTOR_SIZE = 8;

    typedef float vec
	__attribute__ ((vector_size (sizeof(float) * VECTOR_SIZE)));

    vec temp = {0};

    N /= VECTOR_SIZE;

    vec *Av = (vec *)A;
    vec *Bv = (vec *)B;

    for(int i = 0; i < N; ++i) {
	temp += *Av * *Bv;

	Av++;
	Bv++;
    }

    union {
	vec tempv;
	float tempf[VECTOR_SIZE];
    };

    tempv = temp;

    float dot = 0;
    for(int i = 0; i < VECTOR_SIZE; ++i) {
	dot += tempf[i];
    }

    return dot;
}

float avx_unroll_dot(int N, float *A, float *B) {
    const int VECTOR_SIZE = 8;

    typedef float vec
	__attribute__ ((vector_size (sizeof(float) * VECTOR_SIZE)));

    vec temp1 = {0}, temp2 = {0};

    N /= VECTOR_SIZE * 2;

    vec *Av = (vec *)A;
    vec *Bv = (vec *)B;

    for(int i = 0; i < N; ++i) {
	temp1 += *Av * *Bv;

	Av++;
	Bv++;

	temp2 += *Av * *Bv;

	Av++;
	Bv++;
    }

    union {
	vec tempv;
	float tempf[VECTOR_SIZE];
    };

    tempv = temp1;

    float dot = 0;
    for(int i = 0; i < VECTOR_SIZE; ++i) {
	dot += tempf[i];
    }

    tempv = temp2;

    for(int i = 0; i < VECTOR_SIZE; ++i) {
	dot += tempf[i];
    }

    return dot;
}

float avx_unroll_prefetch_dot(int N, float *A, float *B) {
    const int VECTOR_SIZE = 8;

    typedef float vec
	__attribute__ ((vector_size (sizeof(float) * VECTOR_SIZE)));

    vec temp1 = {0}, temp2 = {0};

    N /= VECTOR_SIZE * 2;

    vec *Av = (vec *)A;
    vec *Bv = (vec *)B;

    for(int i = 0; i < N; ++i) {
	__builtin_prefetch(Av + 2, 0, 0);
	temp1 += *Av * *Bv;

	Av++;
	Bv++;

	temp2 += *Av * *Bv;

	Av++;
	Bv++;
    }

    union {
	vec tempv;
	float tempf[VECTOR_SIZE];
    };

    tempv = temp1 + temp2;

    float dot = 0;
    for(int i = 0; i < VECTOR_SIZE; ++i) {
	dot += tempf[i];
    }

    return dot;
}
