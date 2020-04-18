"""
This script modifies the mnist_*.csv internal arrangement
to be compatible with the OmniLearn csv reader
"""
content = open("mnist_test.csv", "r").readlines()


for i in range(0, len(content)):
  first = content[i].split(",")[0]
  lab = [0,0,0,0,0,0,0,0,0,0]
  lab[int(first)] = 1
  line = ","
  for j in range(len(lab)):
    line = line + "," + str(lab[j])
  line = content[i][2:-1] + line + "\n"
  content[i] = line
  lab[int(first)] = 0

title = str()
for i in range(1, 29):
  for j in range(1, 29):
    title = title + "px" + str(i) + "-" + str(j) + ","

title = title + ","

for i in range(0, 10):
  title = title + str(i) + ","

title = title[:-1] + "\n"

out = open("test.csv", "w")
out.write(title)

for i in range(0, len(content)):
  out.write(content[i])