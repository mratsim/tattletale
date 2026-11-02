// Minimal C11 threads subset for Apple's SDK, which ships no <threads.h>.
// threadpool.h includes <threads.h> for thrd_create and thrd_join only,
// both map onto pthreads here.
#pragma once

#include <stdint.h>
#include <pthread.h>

typedef pthread_t thrd_t;
enum { thrd_success = 0, thrd_error = 1 };

typedef int (*thrd_start_t)(void*);

static inline int thrd_create(thrd_t* thr, thrd_start_t fn, void* arg) {
  return pthread_create(thr, NULL, (void* (*)(void*))fn, arg) == 0 ? thrd_success : thrd_error;
}

// C11 moves the start routine's int return into the void* join value
static inline int thrd_join(thrd_t thr, int* res) {
  void* value = NULL;
  int rc = pthread_join(thr, &value);
  if (res != NULL) {
    *res = (int)(intptr_t)value;
  }
  return rc == 0 ? thrd_success : thrd_error;
}
