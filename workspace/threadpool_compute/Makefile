# threadpool_compute build
#
# Everything compiles as one translation unit: src/threadpool_runtime.c
# includes every source and selects the platform files by preprocessor,
# so no uname logic lives here. Out of tree: one object in build/,
# the static library in build/, test and bench binaries beside it.

CC       ?= cc
AR       ?= ar
CFLAGS   ?= -O2 -std=c11 -Wall -Wextra -Werror
INCLUDES := -Isrc/threadpool -Iinclude
LDLIBS   := -lpthread
LIB      := build/libthreadpool_compute.a
RUNTIME  := src/threadpool_runtime.c

TESTS     := test_pool_fib test_parallel_for test_stress test_taskqueues test_deque_conc
BENCHES   := fib nqueens
TEST_BINS := $(addprefix build/,$(TESTS))
BENCH_BINS:= $(addprefix build/bench_,$(BENCHES))
BENCH_CFLAGS := $(filter-out -std=%,$(CFLAGS))

all: $(TEST_BINS)

$(LIB): $(RUNTIME)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o build/threadpool_runtime.o
	$(AR) rcs $@ build/threadpool_runtime.o

$(TEST_BINS): build/%: tests/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB) $(LDLIBS) -o $@

$(BENCH_BINS): build/bench_%: bench/bench_%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) -std=gnu11 $(BENCH_CFLAGS) -Ibench $(INCLUDES) $< $(LIB) $(LDLIBS) -o $@

test: $(TEST_BINS)
	@for t in $(TESTS); do ./build/$$t || exit 1; done

bench: $(BENCH_BINS)

clean:
	rm -rf build

.PHONY: all test bench clean
