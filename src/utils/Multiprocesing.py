from multiprocessing import Process


def print_func(continent='Asia', index=0):
    while True:
        print('The name of continent is : ', continent, " with index: " + str(index) + "\n")


if __name__ == "__main__":  # confirms that the code is under main function
    names = ['America', 'Europe', 'Africa']
    procs = []
    """
    proc = Process(target=print_func)  # instantiating without any argument
    procs.append(proc)
    proc.start()
    """

    # instantiating process with arguments
    for i, name in enumerate(names):
        # print(name)
        proc = Process(target=print_func, args=(name,i))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()