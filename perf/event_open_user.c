#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/time.h>
#include <asm/unistd.h>

#define EVENT_NR 34
#define EVENT_CUR 4

uint64_t INTERVAL;
int ROUND;
char* OUTPUT_FILE;

struct event {
	__u32 event_type;
	__u64 event_config;
};

struct timeval global_time;

struct event event_monitor[EVENT_NR+1];

void event_initialize() {

	event_monitor[0].event_type = PERF_TYPE_HARDWARE;
	event_monitor[0].event_config = PERF_COUNT_HW_INSTRUCTIONS;

	event_monitor[1].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[1].event_config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[2].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[2].event_config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[3].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[3].event_config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[4].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[4].event_config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[5].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[5].event_config = (PERF_COUNT_HW_CACHE_L1D)|(PERF_COUNT_HW_CACHE_OP_PREFETCH << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[6].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[6].event_config = (PERF_COUNT_HW_CACHE_L1I)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[7].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[7].event_config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[8].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[8].event_config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[9].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[9].event_config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[10].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[10].event_config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[11].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[11].event_config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_PREFETCH << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[12].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[12].event_config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_PREFETCH << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[13].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[13].event_config = (PERF_COUNT_HW_CACHE_DTLB)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[14].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[14].event_config = (PERF_COUNT_HW_CACHE_DTLB)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[15].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[15].event_config = (PERF_COUNT_HW_CACHE_DTLB)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[16].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[16].event_config = (PERF_COUNT_HW_CACHE_DTLB)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[17].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[17].event_config = (PERF_COUNT_HW_CACHE_ITLB)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[18].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[18].event_config = (PERF_COUNT_HW_CACHE_ITLB)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[19].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[19].event_config = (PERF_COUNT_HW_CACHE_BPU)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[20].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[20].event_config = (PERF_COUNT_HW_CACHE_BPU)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[21].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[21].event_config = (PERF_COUNT_HW_CACHE_NODE)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[22].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[22].event_config = (PERF_COUNT_HW_CACHE_NODE)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[23].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[23].event_config = (PERF_COUNT_HW_CACHE_NODE)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[24].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[24].event_config = (PERF_COUNT_HW_CACHE_NODE)|(PERF_COUNT_HW_CACHE_OP_WRITE << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[25].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[25].event_config = (PERF_COUNT_HW_CACHE_NODE)|(PERF_COUNT_HW_CACHE_OP_PREFETCH << 8)|(PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);

	event_monitor[26].event_type = PERF_TYPE_HW_CACHE;
	event_monitor[26].event_config = (PERF_COUNT_HW_CACHE_NODE)|(PERF_COUNT_HW_CACHE_OP_PREFETCH << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

	event_monitor[27].event_type = PERF_TYPE_HARDWARE;
	event_monitor[27].event_config = PERF_COUNT_HW_CPU_CYCLES;

	event_monitor[28].event_type = PERF_TYPE_HARDWARE;
	event_monitor[28].event_config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;

	event_monitor[29].event_type = PERF_TYPE_HARDWARE;
	event_monitor[29].event_config = PERF_COUNT_HW_BRANCH_MISSES;

	event_monitor[30].event_type = PERF_TYPE_SOFTWARE;
	event_monitor[30].event_config = PERF_COUNT_SW_PAGE_FAULTS;

	event_monitor[31].event_type = PERF_COUNT_SW_CONTEXT_SWITCHES;
	event_monitor[31].event_config = PERF_COUNT_SW_PAGE_FAULTS;

	event_monitor[32].event_type = PERF_TYPE_HARDWARE;
	event_monitor[32].event_config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;

	event_monitor[33].event_type = PERF_TYPE_HARDWARE;
	event_monitor[33].event_config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
};


uint64_t rdtsc(void) {
	uint64_t a, d;
	__asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
	return (d<<32) | a;
}

uint64_t** time;

int main(int argc, char **argv) {
	struct perf_event_attr pe[EVENT_NR];
	int fd[EVENT_NR];

	if (argc < 5){
		fprintf(stderr, "Invalid number of arguments");
		exit(1);
	}

	int cpuid = atoi(argv[1]);
	INTERVAL = atoi(argv[2])*(uint64_t)1000;
	ROUND = atoi(argv[3]);
	OUTPUT_FILE = argv[4];

	int i, j, k;

	time = (uint64_t**)malloc(ROUND*sizeof(uint64_t*));
	for (j=0; j<ROUND; j++)
		time[j] = (uint64_t*)malloc((EVENT_NR+1)*sizeof(uint64_t));

	event_initialize();

	for (i=0; i<EVENT_NR; i++) {
		memset(&pe[i], 0, sizeof(struct perf_event_attr));
		pe[i].type = event_monitor[i].event_type;
		pe[i].config = event_monitor[i].event_config;
		pe[i].size = sizeof(struct perf_event_attr);
		pe[i].disabled = 1;
		pe[i].inherit = 1;
		pe[i].pinned = 1;
		pe[i].exclude_kernel = 0;
		pe[i].exclude_user = 0;
		pe[i].exclude_hv = 0;
		pe[i].exclude_host = 0;
		pe[i].exclude_guest = 0;

	}

	uint64_t start_cycle;

	int event_index;
	for (j=0; j<ROUND; j++) {
		for (i=0; i<(EVENT_NR/EVENT_CUR); i++) {
			for (k=0; k<EVENT_CUR; k++) {
				event_index = i*EVENT_CUR+k;
				fd[event_index] = syscall(__NR_perf_event_open, &pe[event_index], -1, cpuid, -1, 0);
				if (fd[event_index] == -1) {
					fprintf(stderr, "Error opening leader %llx\n", pe[event_index].config);
					exit(EXIT_FAILURE);
				}

				ioctl(fd[event_index], PERF_EVENT_IOC_RESET, 0);
				ioctl(fd[event_index], PERF_EVENT_IOC_ENABLE, 0);
			}

			start_cycle = rdtsc();
			while(rdtsc()-start_cycle<INTERVAL);

			for (k=0; k<EVENT_CUR; k++) {
				event_index = i*EVENT_CUR+k;
				ioctl(fd[event_index], PERF_EVENT_IOC_DISABLE, 0);
				read(fd[event_index], &time[j][event_index], sizeof(long long));
				close(fd[event_index]);
			}

			// Get time stamp
			gettimeofday(&global_time, NULL);
			time[j][EVENT_NR] = global_time.tv_sec * 1000000 + global_time.tv_usec;
		}

		i = EVENT_NR/EVENT_CUR;
		for (k=0; k<EVENT_NR%EVENT_CUR; k++) {
			event_index = i*EVENT_CUR+k;
			fd[event_index] = syscall(__NR_perf_event_open, &pe[event_index], -1, cpuid, -1, 0);
			if (fd[event_index] == -1) {
				fprintf(stderr, "Error opening leader %llx\n", pe[event_index].config);
				exit(EXIT_FAILURE);
			}

			ioctl(fd[i*EVENT_CUR+k], PERF_EVENT_IOC_RESET, 0);
			ioctl(fd[i*EVENT_CUR+k], PERF_EVENT_IOC_ENABLE, 0);
		}

		start_cycle = rdtsc();
		while(rdtsc()-start_cycle<INTERVAL);

		for (k=0; k<EVENT_NR%EVENT_CUR; k++) {
			event_index = i*EVENT_CUR+k;
			ioctl(fd[event_index], PERF_EVENT_IOC_DISABLE, 0);
			read(fd[event_index], &time[j][event_index], sizeof(long long));
			close(fd[event_index]);
		}
	}

	FILE *output = fopen(OUTPUT_FILE, "w+");
	for (j=0; j<ROUND; j++) {
		for (i=0; i<EVENT_NR+1; i++) {
			fprintf(output, "%lu ", time[j][i]);
		}
		fprintf(output, "\n");
	}

	fclose(output);

}
