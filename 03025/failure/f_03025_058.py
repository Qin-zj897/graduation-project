string = input()
sum_of_o = 0
for i in string:
    if i == 'o':
        sum_of_o += 1

if 15-len(string) > 8 - sum_of_o:
    print("YES")
else:
    print("NO")