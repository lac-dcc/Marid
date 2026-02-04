#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

int16_t foo(bool cond) {
    int16_t a = 1;
    int16_t b = 2;
    int16_t c = 3;

    // sign-extension of b
    int32_t D = (int32_t)b;

    int16_t acc = 0;

    for (size_t i = 0; i < 2; i += 1) {
        int16_t v0;
        if (cond) {
            int16_t v2 = a + c;
            int16_t v3 = (int16_t)D;
            v0 = v2 + v3;
        } else {
            v0 = c + c;
        }
        acc = v0;
    }

    return acc;
}
