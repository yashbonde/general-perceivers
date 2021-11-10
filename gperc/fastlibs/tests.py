from gperc_fast import sum_array, buffered_read

print(sum_array([123, 123, 23, 4, 234, 235]))
sample = ("./cargo.toml", 10, 128)
out = buffered_read(*sample)
print("".join([chr(i) for i in out]))
