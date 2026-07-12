import hid, time
d = hid.device()
d.open(0x256f, 0xc635)
d.set_nonblocking(True)
print("opened:", d.get_product_string())
for i in range(120):
 data = d.read(64)
 if data: print(data)
 time.sleep(0.04)
d.close()
