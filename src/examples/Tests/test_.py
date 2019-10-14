def in_list(iter, n):
    start_index = 0
    end_index = len(iter) - 1
    middle_index = len(iter) // 2
    quarter_index = middle_index // 2
    third_quarter_index = int(middle_index * 1.5)

    section_1 = [start_index, quarter_index]
    section_2 = [quarter_index, middle_index]
    section_3 = [middle_index, third_quarter_index]
    section_4 = [third_quarter_index, end_index]
    sections = [section_1, section_2, section_3, section_4]

    found_n = False
    for section in sections:
        for i in iter[section[0]:section[1]]:
            if i == n:
                found_n = True
                print(True, f"{n} is in {iter[section[0]:section[1]]}")
    if not found_n:
        print(False, f"{n} is NOT in {iter}")

iter = [2,1,5,6,7,3,9,8]
n = 5

in_list(iter=iter, n=n)
