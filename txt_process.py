
input_path = 'over_spa_random_nore_wid32100.out'
out_path = 'over_spa_random_nore_wid32100_777.out'
with open(input_path, 'r', encoding='utf-8') as inf:
    count = 0
    lines_list = []
    for line in inf.readlines():
        count += 1
        if count < 224532:
            lines_list.append(line)
with open(out_path, 'w', encoding='utf-8') as outf:
    for line in lines_list:
        outf.write(line)
