import os

for root, dirs, files in os.walk(".", topdown=True):
    for name in files:
        if name == "music.db":
            full = os.path.join(root, name)
            size = os.path.getsize(full)
            print(full, " --> size:", size, "bytes")
