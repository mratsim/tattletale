// Minimal C11 threads subset for Apple's SDK, which ships no <threads.h>.
// threadpool.h includes <threads.h> for thrd_create and thrd_join only,
// both map onto pthreads here.
#pragma once

#if defined(__APPLE__)

#include <pthread.h>

typedef pthread_t thrd_t;
enum { thrd_success = 0, thrd_error = 1 };

typedef int (*thrd_start_t)(void*);

static inline int thrd_create(thrd_t* thr, thrd_start_t fn, void* arg) {
  return pthread_create(thr, NULL, (void* (*)(void*))fn, arg) == 0 ? thrd_success : thrd_error;
}

static inline int thrd_join(thrd_t thr, void** ret) {
  void* value = NULL;
  int rc = pthread_join(thr, &value);
  if (ret != NULL) {
    *ret = value;
  }
  return rc == 0 ? thrd_success : thrd_error;
}

#endif
