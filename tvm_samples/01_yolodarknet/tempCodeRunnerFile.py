
lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")
darknet_lib = __darknetffi__.dlopen(lib_path)
net = darknet_lib.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
