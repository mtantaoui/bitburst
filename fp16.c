#include <immintrin.h>
#include <stdio.h>

// This function adds two arrays of 16 half-precision floats
void add_half_precision_vectors(__fp16 *a, __fp16 *b, __fp16 *result)
{
    __m256h va = _mm256_loadu_ph(a); // Load 16 half-precision floats
    __m256h vb = _mm256_loadu_ph(b);
    __m256h vsum = _mm256_add_ph(va, vb); // Perform addition
    _mm256_storeu_ph(result, vsum);       // Store result
}

int main()
{
    __fp16 a[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    __fp16 b[16] = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0,
                    8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    __fp16 result[16];

    add_half_precision_vectors(a, b, result);

    for (int i = 0; i < 16; i++)
    {
        printf("result[%d] = %f\n", i, (float)result[i]);
    }

    return 0;
}
