for name in ['capa.txt', 'capa1.txt', 'capa2.txt', 'capb.txt', 'capb1.txt', 'capc.txt', 'capc1.txt']:
    path = f"data/raw/{name}"
    print(f"=== {name} ===")
    with open(path, 'r') as f:
        for i in range(15):
            print(f.readline().strip())
