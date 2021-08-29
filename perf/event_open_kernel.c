#include <linux/module.h>
#include <linux/syscalls.h>
#include <linux/delay.h>

static uint64_t SYS_CALL_TABLE_ADDR = 0xffffffff81801360;

asmlinkage int (*kernel_call_perf_event_open)(struct perf_event_attr *attr_uptr, pid_t pid, int cpu, int group_fd, unsigned long flags);
asmlinkage int (*kernel_call_close)(unsigned int fd);
asmlinkage int (*kernel_call_ioctl)(unsigned int fd, unsigned int cmd, unsigned long arg);
asmlinkage int (*kernel_call_read)(unsigned int fd, long long *buf, size_t count);

void ** kernel_call_sys_call_table;

static int __init kernel_syscall_init(void)
{
    struct perf_event_attr pe;
    int fd;
    int ret;
    long long count;
    mm_segment_t fs;

    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    pe.disabled = 1;
    pe.exclude_kernel = 0;
    pe.exclude_hv = 1;

    kernel_call_sys_call_table = (void**) SYS_CALL_TABLE_ADDR;
    kernel_call_perf_event_open = kernel_call_sys_call_table[__NR_perf_event_open];
    kernel_call_ioctl = kernel_call_sys_call_table[__NR_ioctl];
    kernel_call_close = kernel_call_sys_call_table[__NR_close];
    kernel_call_read = kernel_call_sys_call_table[__NR_read];


    fs = get_fs();
    set_fs (get_ds());
    fd = kernel_call_perf_event_open(&pe, 15796, -1, -1, 0);
    set_fs(fs);
    if (fd == -1) {
        printk("open perf_event errors!\n");
        return -1;
    }

    ret = kernel_call_ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    if (ret == -1) {
        printk("reset counter errors!\n");
        return -1;
    }

    ret = kernel_call_ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    if (ret == -1) {
        printk("enable counter errors!\n");
        return -1;
    }

    printk("measuring instruction counter!\n");

    msleep(5000);

    ret = kernel_call_ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    if (ret == -1) {
        printk("disable counter errors!\n");
        return -1;
    }

    fs = get_fs();
    set_fs (get_ds());
    ret = kernel_call_read(fd, &count, sizeof(long long));
    set_fs(fs);
    if (ret == -1) {
        printk("read counter errors!\n");
        return -1;
    }
    printk("Used %lld instructions\n", count);

    ret = kernel_call_close(fd);
    if (ret == -1) {
        printk("close file errors!\n");
        return -1;
    }

    return 0;
}

static void __exit kernel_syscall_exit(void)
{
    printk("Leaving the example module\n");
}

module_init(kernel_syscall_init);
module_exit(kernel_syscall_exit);

MODULE_LICENSE("GPL");
