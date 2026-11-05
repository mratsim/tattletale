/* threadpool_compute: flow events over the C port of weave's flow_events.nim. */

#include "flow_events.h"

#include <stdio.h>
#include <stdlib.h>

/* A task stack stands in for the runtime's ready queue.
 * Trigger's enqueue callback pushes drained tasks onto it.
 * tpc_task's parent field links the stack and the test owns the tasks.
 */
typedef struct {
    tpc_task* top;
    int count;
} TaskStack;

static void stack_add(void* queue, tpc_task* task) {
    TaskStack* stack = (TaskStack*)queue;
    task->parent = stack->top;
    stack->top = task;
    stack->count++;
}

static tpc_task* allocTask(void) {
    tpc_task* task = (tpc_task*)calloc(1, sizeof(tpc_task));
    if (task == NULL) {
        fprintf(stderr, "task allocation failed\n");
        exit(1);
    }
    return task;
}

static int expect(int cond, const char* what) {
    if (!cond) {
        fprintf(stderr, "flow events test failed: %s\n", what);
        return 1;
    }
    return 0;
}

/* Port of weave's mainSingle self-test: one delayed task, trigger,
 * one late task that schedules itself.
 */
static int mainSingle(void) {
    TaskStack stack = {NULL, 0};

    tpc_flow_event event1;
    tpc_flow_event_initSingle(&event1);
    tpc_task* task = allocTask();
    if (expect(tpc_flow_event_delayedUntil(task, event1), "single: task not delayed")) {
        return 1;
    }
    if (expect(stack.count == 0, "single: stack nonempty before trigger")) {
        return 1;
    }

    tpc_flow_event_trigger(event1, &stack, stack_add);
    if (expect(stack.count == 1, "single: trigger did not schedule the task")) {
        return 1;
    }

    tpc_task* late = allocTask();
    if (expect(!tpc_flow_event_delayedUntil(late, event1), "single: late task delayed")) {
        return 1;
    }
    if (expect(stack.count == 1, "single: late task enqueued by the event")) {
        return 1;
    }
    stack_add(&stack, late);
    if (expect(stack.count == 2, "single: late task not enqueueable")) {
        return 1;
    }

    tpc_flow_event event2;
    tpc_flow_event_initSingle(&event2);
    tpc_task* task2a = allocTask();
    tpc_task* task2b = allocTask();
    if (expect(tpc_flow_event_delayedUntil(task2a, event2), "single: task2a not delayed")) {
        return 1;
    }
    if (expect(tpc_flow_event_delayedUntil(task2b, event2), "single: task2b not delayed")) {
        return 1;
    }
    if (expect(stack.count == 2, "single: stack changed before second trigger")) {
        return 1;
    }
    tpc_flow_event_trigger(event2, &stack, stack_add);
    if (expect(stack.count == 4, "single: trigger did not drain both tasks")) {
        return 1;
    }

    tpc_flow_event_unref(&event1);
    tpc_flow_event_unref(&event2);
    printf("Simple event: SUCCESS\n");
    return 0;
}

/* Port of weave's mainLoop self-test: one delayed iteration task,
 * trigger of iteration 3, one late task that schedules itself.
 */
static int mainLoop(void) {
    TaskStack stack = {NULL, 0};

    tpc_flow_event event1;
    tpc_flow_event_initIter(&event1, 0, 10, 1);
    tpc_task* task = allocTask();
    if (expect(tpc_flow_event_delayedUntilIter(task, event1, 3), "loop: task not delayed")) {
        return 1;
    }
    if (expect(stack.count == 0, "loop: stack nonempty before trigger")) {
        return 1;
    }

    tpc_flow_event_triggerIter(event1, 3, &stack, stack_add);
    if (expect(stack.count == 1, "loop: trigger did not schedule the task")) {
        return 1;
    }

    tpc_task* late = allocTask();
    if (expect(!tpc_flow_event_delayedUntilIter(late, event1, 3), "loop: late task delayed")) {
        return 1;
    }
    if (expect(stack.count == 1, "loop: late task enqueued by the event")) {
        return 1;
    }
    stack_add(&stack, late);

    tpc_flow_event event2;
    tpc_flow_event_initIter(&event2, 0, 10, 1);
    tpc_task* task2a = allocTask();
    tpc_task* task2b = allocTask();
    if (expect(tpc_flow_event_delayedUntilIter(task2a, event2, 4), "loop: task2a not delayed")) {
        return 1;
    }
    if (expect(tpc_flow_event_delayedUntilIter(task2b, event2, 4), "loop: task2b not delayed")) {
        return 1;
    }
    if (expect(stack.count == 2, "loop: stack changed before second trigger")) {
        return 1;
    }
    tpc_flow_event_triggerIter(event2, 4, &stack, stack_add);
    if (expect(stack.count == 4, "loop: trigger did not drain both tasks")) {
        return 1;
    }

    tpc_flow_event_unref(&event1);
    tpc_flow_event_unref(&event2);
    printf("Loop event: SUCCESS\n");
    return 0;
}

/* Multi-dependency: a task deferred on two single events only schedules
 * after the second trigger. The chain re-defers on the first.
 */
static int mainMulti(void) {
    TaskStack stack = {NULL, 0};

    tpc_flow_event eventA;
    tpc_flow_event eventB;
    tpc_flow_event_initSingle(&eventA);
    tpc_flow_event_initSingle(&eventB);
    tpc_flow_event_dep deps[2] = {
        {eventA, TPC_NO_ITER},
        {eventB, TPC_NO_ITER},
    };

    tpc_task* task = allocTask();
    if (expect(tpc_flow_event_delayedUntilMulti(task, deps, 2), "multi: task not delayed")) {
        return 1;
    }

    tpc_flow_event_trigger(eventA, &stack, stack_add);
    if (expect(stack.count == 0, "multi: task scheduled after one of two triggers")) {
        return 1;
    }

    tpc_flow_event_trigger(eventB, &stack, stack_add);
    if (expect(stack.count == 1, "multi: task not scheduled after both triggers")) {
        return 1;
    }

    tpc_flow_event_unref(&eventA);
    tpc_flow_event_unref(&eventB);
    printf("Multi event: SUCCESS\n");
    return 0;
}

/* Multi-dependency over an iteration event: the task re-defers on bucket 5
 * of the iteration event until that iteration triggers.
 */
static int mainMultiIter(void) {
    TaskStack stack = {NULL, 0};

    tpc_flow_event eventA;
    tpc_flow_event eventIter;
    tpc_flow_event_initSingle(&eventA);
    tpc_flow_event_initIter(&eventIter, 0, 10, 1);
    tpc_flow_event_dep deps[2] = {
        {eventA, TPC_NO_ITER},
        {eventIter, 5},
    };

    tpc_task* task = allocTask();
    if (expect(tpc_flow_event_delayedUntilMulti(task, deps, 2), "multiIter: task not delayed")) {
        return 1;
    }

    tpc_flow_event_trigger(eventA, &stack, stack_add);
    if (expect(stack.count == 0, "multiIter: task scheduled before the iteration fired")) {
        return 1;
    }

    tpc_flow_event_triggerIter(eventIter, 5, &stack, stack_add);
    if (expect(stack.count == 1, "multiIter: task not scheduled after the iteration fired")) {
        return 1;
    }

    tpc_flow_event_unref(&eventA);
    tpc_flow_event_unref(&eventIter);
    printf("Multi iteration event: SUCCESS\n");
    return 0;
}

int main(void) {
    if (mainSingle() != 0) {
        return 1;
    }
    if (mainLoop() != 0) {
        return 1;
    }
    if (mainMulti() != 0) {
        return 1;
    }
    if (mainMultiIter() != 0) {
        return 1;
    }
    return 0;
}
