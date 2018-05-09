
from __future__ import division

import psutil

from subprocess import PIPE, Popen
import psutil

def cpu_temperature():
    process = Popen(['vcgencmd', 'measure_temp'], stdout=PIPE)
    output, _error = process.communicate()
    output_str = str(output)
    return float(output_str[output_str.index('=') + 1:output_str.rindex("'")])

def ram_info():
    """
    TODO (@Gabvaztor) Return the percent usage.
    Returns:

    """
    ram_information = psutil.virtual_memory()
    return str(ram_information)

def cpu_usage():
    cpu_usage = psutil.cpu_percent()
    cpu_usage2 = psutil.cpu_percent(interval=1)
    cpu_usage3 = psutil.cpu_percent(interval=1, percpu=True)
    print("CPU usage: ", cpu_usage)
    print("CPU usage 2: ",cpu_usage2)
    print("CPU usage 3: ",cpu_usage3)
    return cpu_usage2

def disk_usage():
    disk = psutil.disk_usage('/')
    disk_total = disk.total / 2 ** 30  # GiB.
    disk_used = disk.used / 2 ** 30
    disk_free = disk.free / 2 ** 30
    disk_percent_used = disk.percent
    print("Total Disk Mem: ", disk_total)
    print("Total Disk Used: ", disk_used)
    print("Total Disk Free Mem: ", disk_free)
    print("Percent Disk Mem Used: ", disk_percent_used)
    return disk_percent_used

def main():
    cpu_temp = cpu_temperature()
    cpu_usage = psutil.cpu_percent()
    cpu_usage2 = psutil.cpu_percent(interval=1)
    cpu_usage3 = psutil.cpu_percent(interval=1, percpu=True)

    """
    ram = psutil.phymem_usage()
    ram_total = ram.total / 2 ** 20  # MiB.
    ram_used = ram.used / 2 ** 20
    ram_free = ram.free / 2 ** 20
    ram_percent_used = ram.percent
    print(ram_total)
    print(ram_used)
    print(ram_free)
    print(ram_percent_used)
    """
    disk = psutil.disk_usage('/')
    disk_total = disk.total / 2 ** 30  # GiB.
    disk_used = disk.used / 2 ** 30
    disk_free = disk.free / 2 ** 30
    disk_percent_used = disk.percent

    print("CPU Temp: ", cpu_temp)
    print("CPU usage: ", cpu_usage)
    print("CPU usage 2: ",cpu_usage2)
    print("CPU usage 3: ",cpu_usage3)
    print("RAM stats: ", psutil.virtual_memory())
    print("Total Disk Mem: ", disk_total)
    print("Total Disk Used: ", disk_used)
    print("Total Disk Free Mem: ", disk_free)
    print("Percent Disk Mem Used: ", disk_percent_used)

    # Show SO process
    processes = [(p.memory_info().vms, p) for p in psutil.process_iter()]
    print(str(processes))
    # Print top five processes in terms of virtual memory usage.
    for virtual_memory, process in processes[:5]:
        print(virtual_memory // 2 ** 20, process.pid, process.name)

main()