f = open("test.py", 'r', encoding="utf-8")
while True:
    line = f.readline()
    if not line: break
    print(line)
f.close()

f = open('myFile.txt', 'w')
f.write('This is my first file.')
f.close


w = open('myFile.txt', 'r')
file_txt = w.read()
w.close()
print(file_txt)